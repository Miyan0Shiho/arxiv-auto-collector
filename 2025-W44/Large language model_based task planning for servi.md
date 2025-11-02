# Large language model-based task planning for service robots: A review

**Authors**: Shaohan Bian, Ying Zhang, Guohui Tian, Zhiqiang Miao, Edmond Q. Wu, Simon X. Yang, Changchun Hua

**Published**: 2025-10-27 14:06:40

**PDF URL**: [http://arxiv.org/pdf/2510.23357v1](http://arxiv.org/pdf/2510.23357v1)

## Abstract
With the rapid advancement of large language models (LLMs) and robotics,
service robots are increasingly becoming an integral part of daily life,
offering a wide range of services in complex environments. To deliver these
services intelligently and efficiently, robust and accurate task planning
capabilities are essential. This paper presents a comprehensive overview of the
integration of LLMs into service robotics, with a particular focus on their
role in enhancing robotic task planning. First, the development and
foundational techniques of LLMs, including pre-training, fine-tuning,
retrieval-augmented generation (RAG), and prompt engineering, are reviewed. We
then explore the application of LLMs as the cognitive core-`brain'-of service
robots, discussing how LLMs contribute to improved autonomy and
decision-making. Furthermore, recent advancements in LLM-driven task planning
across various input modalities are analyzed, including text, visual, audio,
and multimodal inputs. Finally, we summarize key challenges and limitations in
current research and propose future directions to advance the task planning
capabilities of service robots in complex, unstructured domestic environments.
This review aims to serve as a valuable reference for researchers and
practitioners in the fields of artificial intelligence and robotics.

## Full Text


<!-- PDF content starts -->

Large language model-based task planning for service robots: A review
Shaohan Biana, Ying Zhanga,b,c,∗, Guohui Tiand, Zhiqiang Miaoe, Edmond Q. Wuf, Simon X. Yangg
and Changchun Huaa,b,c
aSchool of Electrical Engineering, Yanshan University, Qinhuangdao 066004, China
bEngineering Research Center of Intelligent Control System and Intelligent Equipment, Ministry of Education, Yanshan University, Qinhuangdao
066004, China
cHebei Key Laboratory of Intelligent Rehabilitation and Neuromodulation, Yanshan University, Qinhuangdao 066004, China
dSchool of Control Science and Engineering, Shandong University, Jinan 250061, China
eNational Engineering Research Center of Robot Visual Perception and Control Technology, Hunan University, Changsha 410082, China
fDepartment of Automation, Shanghai Jiao Tong University, Shanghai 200240, China
gAdvanced Robotics and Intelligent Systems Laboratory, School of Engineering, University of Guelph, Guelph N1G 2W1, Canada
ARTICLE INFO
Keywords:
Large language model
Service robot
Task planning
ReviewABSTRACT
With the rapid advancement of large language models (LLMs) and robotics, service robots are
increasingly becoming an integral part of daily life, offering a wide range of services in complex
environments.Todelivertheseservicesintelligentlyandefficiently,robustandaccuratetaskplanning
capabilities are essential. This paper presents a comprehensive overview of the integration of LLMs
into service robotics, with a particular focus on their role in enhancing robotic task planning.
First, the development and foundational techniques of LLMs, including pre-training, fine-tuning,
retrieval-augmented generation (RAG), and prompt engineering, are reviewed. We then explore
the application of LLMs as the cognitive core—“brain”—of service robots, discussing how LLMs
contribute to improved autonomy and decision-making. Furthermore, recent advancements in LLM-
driven task planning across various input modalities are analyzed, including text, visual, audio, and
multimodal inputs. Finally, we summarize key challenges and limitations in current research and
propose future directions to advance the task planning capabilities of service robots in complex,
unstructureddomesticenvironments.Thisreviewaimstoserveasavaluablereferenceforresearchers
and practitioners in the fields of artificial intelligence and robotics.
1. Introduction
With the continuous progress of robotics [1, 2] and
artificial intelligence (AI) [3, 4], the perception, reasoning,
and action capabilities of service robots have been signifi-
cantlyenhanced.Asaresult,servicerobotsareevolvinginto
capable assistants within domestic environments [5, 6, 7].
By autonomously performing various household tasks such
asdesktoporganizing,floorcleaningandevencaringforthe
elderly, these robots contribute to improving the quality of
life [8, 9].
To effectively accomplish daily tasks, service robots
mustengageindetailedtaskplanning,whichinvolvesunder-
standing user requirements, decomposing high-level goals
into subtasks, and generating executable action sequences.
Task planning is the core technology to realize autonomous
decision-makingofservicerobots,whichenablestherobots
toanalyzeenvironmentalinformation,evaluatetaskrequire-
ments, and develop optimal action plans. Such autonomous
decision-making capability is the key to robot intelligence.
Traditionalrobotictaskplanningmethodsmainlyrelyonde-
tailedtaskmodelsandenvironmentinformation[10,11,12].
For example, the task planning problem is typically divided
into the domain description and problem description [13],
wherethedomaindescribesasetofactionsdefinedbytheir
preconditions and subsequent effects, while the problem
description specifies the initial state and the desired goal
∗Corresponding author.
yzhang@ysu.edu.cn(Y. Zhang)conditions.However,suchapproachesgenerallypresuppose
finiteness, determinism, and invariance of planning goals.
These assumptions limit their applicability, especially in
real-world scenarios where environments are dynamic and
unpredictable. Moreover, traditional methods often suffer
fromlowfaulttolerance,limitedscalability,andweakadapt-
ability to real-world interactions. In real-world domestic
environments, the positional relationship between objects
andscenesisusuallydynamicallychanging[14].Challenges
suchasclutter,occlusionbetweenindividualobjectsfurther
lead to an increase in the perception difficulty of service
robots[15].Inaddition,duetothelackofaprioriknowledge
oftheenvironmentandthetargetobject,theefficiencyoftar-
getsearchinlarge-scalescenariostendstobelow,andblind
search of the entire home environment is time-consuming,
labor-intensive and difficult to apply to real life [16, 17].
Therefore, how to generate executable task planning that
meets user needs in complex and dynamically changing
environments is one of the critical challenges in deploying
service robots in unstructured domestic environments.
Inrecentyears,withthecontinuousdevelopmentoflarge
language models (LLMs), they have shown great superi-
ority and adaptability in various fields. These giant neural
networks with tens of billions of parameters have shown
unprecedented comprehension and generation capabilities
through training on massive amounts of data. From the
stunningperformanceofChatGPTtothemultimodalbreak-
through of GPT-4, LLMs are redefining the boundaries of
machine intelligence. The deep fusion of LLM and robotics
First Author et al.:Preprint submitted to ElsevierPage 1 of 24arXiv:2510.23357v1  [cs.RO]  27 Oct 2025

LLM-based task planning for service robots: A review
[18] is driving the evolution of robots from single-task
performers to general-purpose intelligences. With their rich
common-sense knowledge and reasoning ability as the cog-
nitive core of the intelligences, these models have dramat-
ically improved the service robot’s ability to understand
and adapt to the environment. And through the fusion pro-
cessing of multimodal data such as vision and speech, the
robot is able to more accurately perceive and understand
the surrounding environment, providing a reliable basis for
autonomousdecision-making.LLMsshowevenmoreamaz-
ing potential in task planning for service robots. They are
able to break down abstract instructions into concrete steps,
consider environmental constraints, and formulate reason-
able action plans. This ability allows the robots to be no
longer limited to preset programs, but to respond flexibly to
a variety of complex tasks, understand user needs in depth,
and even provide personalized services.
Recent literature has seen a surge in reviews exploring
theintegrationofLLMswithrobotics.Kimetal.[19]offered
a foundational analysis of LLM-enhanced robotic systems,
covering communication, perception, and control. Zeng et
al. [20] and Li et al. [21] delved into specific applications,
respectively examining LLMs’ impact on decision-making
and path planning, and their versatility in multi-robot coor-
dination.Cuietal.[22]tookauniquetask-centricapproach,
investigatingautonomoustaskdiscoverythroughafusionof
visual semantics and LLMs. Furthermore, several reviews
have focused on narrower, yet critical, subfields. These in-
clude surveys on traditional robotics topics like navigation
and perception [23, 24, 25], as well as those centered on
advancedLLM-drivenfunctionalitiessuchasreasoningand
tool use [26, 27, 28].
While existing surveys have comprehensively covered
variousaspectsofroboticsandLLMintegration,ourreview
identifies a critical gap in systematic analysis of task plan-
ning specifically for domestic service robots. This domain
presents unique challenges due to the unstructured nature
ofhomeenvironmentsanddiverseuserinteractionpatterns,
which demand specialized approaches not fully addressed
in prior works. Unlike previous reviews that broadly survey
LLM applications in robotics, our work not only provides
a concise and systematic overview of the fundamentals and
core techniques of LLMs, emphasizing their application to
task planning for service robots operating in home envi-
ronments, but also pioneers a modality-centric taxonomy
(text,vision,audio,multimodal)foranalyzingtaskplanning
challenges, enabling more targeted technological develop-
ment. This novel framework reveals previously overlooked
intermodal synergies and implementation barriers specific
to home service scenarios.
The key innovations and contributions of this paper are
as follows:
•Wepresentasystematicreviewofthedevelopmentof
LLMs and provide a concise overview of their core
techniques, which include pre-training, fine-tuning,
retrieval-augmentedgeneration,andpromptengineer-
ing.•We establish the first modality-centric taxonomy for
analyzing LLM-based task planning in domestic ser-
vice robotics, thereby revealing unique cross-modal
challenges.
•We critically examine the key obstacles hindering
real-world deployment of LLM-based service robots
and propose actionable research pathways to advance
the field.
The rest of the paper is organized as follows. In Section
2, we first introduce the important theoretical foundations
of the LLM, and in Section 3 we provide a systematic
categorizationofLLM-basedtaskplanningforhomeservice
robots based on the differences in the input modalities, and
provide a detailed overview of each category in Sections
4 to 7. In Section 8, we present some challenges to the
current LLM-based robotic task planning and an outlook to
the future. Finally, we conclude the paper in Section 9.
2. Theory Foundation of LLM
In this section, we provide an overview of the theo-
retical foundations of LLM, covering the developmental
backgroundofLLM,thekeytechniques-Pre-training,Fine-
tuning,RAG,andPromptEngineering.Inaddition,wepro-
vide an overview of the application of LLM to robotic task
planning, showing how it empowers robots to achieve more
efficient and intelligent task execution.
2.1. Background
SincetheproposaloftheTransformarchitecture,thede-
velopmentofLLMhasgonethroughtwocorebreakthrough
phases, which have driven the vigorous development of the
natural language processing (NLP) field. The first phase is
marked by the proposal of Transformer [29], whose self-
attention mechanism enables parallelized long sequence
modeling. GPT-1 [30] and [31] adopt unidirectional and
bidirectional pre-training, respectively, to lay down the
two paradigms of generation and comprehension, and the
pre-training-fine-tuning framework achieves significant im-
provement on several NLP tasks. The second phase focuses
on model scale-up and interaction capabilities, with GPT-
3 [32] and ChatGPT [33] demonstrating the potential of
generative AI and optimized dialog interactions, respec-
tively.Currentresearchtrendsareevolvingtowardsefficient,
autonomous intelligences and ethical alignment, gradually
approaching the boundary of general-purpose artificial in-
telligence (AGI). Fig. 1 illustrates the evolution of LLMs
since 2019, and we describe some of these typical LLMs
below.
T5[34] improves on the traditional Transformer model
by moving layer normalization outside of residual con-
nections. It employs masked language modeling as a pre-
training target, using spanwise masking of consecutive to-
kens instead of independent masking of individual tokens,
which shortens sequence length and accelerates training.
Once pre-training is complete, an adapter layer [35] is used
to fine-tune for downstream tasks.
First Author et al.:Preprint submitted to ElsevierPage 2 of 24

LLM-based task planning for service robots: A review
2020 PanGu -a
 Codex
 ERNIE 3.0
 Jurassic -1
 Yuan 1.0
 T0
 Gopher
 WebGPT
 GLaM
 LaMAD MT-NLG
 AlphaCode
 CodeGen
 PaLM
 UL2
 TK-Instruct
 Sparrow
 GLM
 OPT
 U-PALM
 BLOOM
 ChatGPT LLaMA
 Alpaca
 PanGu -S
 GPT -4
 HuaTuo
 Koala
 PaLM 2
 Wizard -LM
 MPT
 StarCoder
 LaMA 2
 Gemini
 Bard
2019 T5 GPT -3
 mT5
2021 2022 2023 Deepseek
 Nemotron
 Gemini -1.5
 Gemma 2
 Command R
 Grok -1
 Grok -1.5
 Mixtral
 Stable LM
 LLaMA 3
 GLM -4
 Qwen 2.5
 DeepSeek -V2
 GPT -4o
 Mistral Large 2
 OpenAI o 1
 Falcon 3
2024 Claude Haiku
 DeepSeek -R1
 Granite 3.2
 Granite Vision
 Grok 3
2025
Figure 1:The evolution of LLM since 2019
GPT-3[32]continuesthearchitectureofGPT-2[30]and
drawsonthedesignofSparseTransformers[36],combining
dense and sparse attention mechanisms in the transformer
layer and employing a gradient noise scale [37] during
training.GPT-3extendsthemodelparametersto175billion,
further validating the positive correlation between model
size and performance improvement.
GLaM[38] represents a family of language models
employing a sparsely activated Mixed (MoE) structure of
decoder experts [39]. GLaM sparsely activates the experts,
with each input token being processed by only the optimal
twoexperts.Themodelkeepsthetrainingenergyconsump-
tion at one-third of GPT-3 while maintaining high perfor-
mance.
GLM-130B[40]isabilingual(English-Chinese)model,
whichisdistinguishedfromtheunidirectionalGPT-3[32]by
the use of autoregressive masks to populate the pre-training
targets for training and the application of gradient contrac-
tion of the embedding layer to ensure training stability.
DeepSeek-v2[41]isaMoEmodelthatintroducesMul-
tihead Latent Attention (MLA) to reduce the inference cost
bycompressingthekey-value(KV)cacheintopotentialvec-
tors.DuetoMLA,theinferencethroughputofDeepSeek-v2
is 5.76 times faster than DeepSeek [42].
2.2. Pre-training
The pre-training of LLM is the most critical basic stage
inthewholemodelconstructionprocess,andthisprocessis
essentially a self-supervised learning approach that allows
the model to automatically extract linguistic laws, construct
a knowledge system, and form a deep understanding of
natural language from massive unlabeled text data. Modern
LLM pre-training is mainly built on the Transformer archi-
tecture[43],andaccordingtodifferentapplicationscenarios
and functional requirements, researchers have developedthree main architectural variants: causal decoder architec-
ture,encoder-decoderarchitecture,andprefixdecoderarchi-
tecture. Although these architectures differ in their specific
implementations, they all rely on Transformer’s core com-
ponent, the self-attention mechanism, to achieve effective
modelingoflong-distancedependenciesthroughtechniques
suchasmulti-attention,residualconnectivity,andlayernor-
malization.
Pre-training task design is the core innovation of LLM
pre-training.Autoregressivelanguagemodeling(e.g.,GPT)
requires the model to predict the next token based on the
above, which perfectly fits the text generation and is the
classic pre-training paradigm. In addition, improvement
schemes such as continuous fragment masking used by
SpanBERT [44] and unified language modeling used by
UniLM [45] enhance the pre-training effect to varying
degrees. In addition the model training phase faces the
challengeofoptimizingtheparametersanddataonanultra-
large scale. Researchers have developed optimizers such as
AdamW [46] and AdaFactor [47] for large-scale training.
The combined use of data parallelism, model parallelism,
and pipeline parallelism substantially improves the training
efficiency. Techniques such as mixed-precision training and
gradientcheckpointingeffectivelyreducethevideomemory
occupation, making the training process more efficient.
2.3. Fine-tuning
The advantages of fine-tuning are its efficiency and per-
formanceimprovement.Comparedtotrainingfromscratch,
fine-tuning greatly saves computational resources and time.
Pre-trained LLMs are good at predicting textual tokens, but
maybelimitedingeneratingstructuredoutputorprocessing
domain-specificinformation.Toovercometheselimitations,
theoutputlayeroftheLLMcanbetunedbyFine-Tuningto
fit specific task requirements. To address the lack of knowl-
edge in LLM, fine-tuning training using domain-specific
data can enhance LLM’s understanding and processing in
First Author et al.:Preprint submitted to ElsevierPage 3 of 24

LLM-based task planning for service robots: A review
Table 1
Comparison of typical LLMs
Model Parameters Advantages Application Areas
T5 [34] 11BMoves layer normalization outside of resid-
ual connections, uses spanwise masking,
and employs an adapter layer for fine-
tuning.Text classification, question answering,
summarization.
GPT-3 [32] 175BCombinesdenseandsparseattentionmech-
anisms, employs a gradient noise scale, and
scales up to 175 billion parameters.Text generation, language understanding,
code generation.
GLaM [38] 1200BEmploys a sparsely activated MoE struc-
ture, processes each input token by only the
optimal two experts, and reduces training
energy consumption.Text generation, language understanding.
GLM-130B [40] 130BUses autoregressive masks for pre-training
targets and applies gradient contraction of
the embedding layer for training stability.Bilingualtextgeneration(English-Chinese),
language understanding.
DeepSeek-v2 [41] 236BIntroduces Multihead Latent Attention
(MLA) to compress the key-value (KV)
cache into potential vectors, significantly
reducing inference cost.Efficient text generation, language under-
standing.
the domain and improve its accuracy and performance on
tasks in the domain.
From the perspective of technical implementation, fine-
tuning is mainly divided into two major directions, Full
Fine-tuning (FFT) [48] and Parameter-Efficient Fine-tuning
(PEFT) [49]. FFT updates all the parameters of the model,
which is effective but requires a large amount of compu-
tational resources and training data, in contrast, PEFT [50,
35, 51] drastically reduces the resource requirements while
guaranteeingtheperformancebyadjustingonlysomeofthe
parameters or by adding small adaptation modules. In addi-
tion,InstructionFine-tuning[52]isanimportantfine-tuning
paradigm that has emerged in recent years, which shapes
the behavioral patterns of a model by providing it with
explicit examples of instructions. Reinforcement Learning
with Human Feedback (RLHF) [53] further enhances the
effectiveness of Instruction Fine-tuning by introducing an
artificial scoring mechanism to make the model outputs
more consistent with human values and preferences.
2.4. Prompt Engineering
PromptEngineeringisakeytechniqueinLLMapplica-
tions that does not require updating the model’s parameters
and aims to guide the model to generate more accurate
and reliable outputs by carefully designing input prompts.
A well designed prompt can significantly improve model
performance, reduce illusions, and adapt to different task
requirements. In the following, we will discuss commonly
used prompt methods.
Instruction Prompting [54] aims to provide the LLM
withexamplesofinstructionpromptssothatitcaneliminate
training or testing discrepancies and simulate real-worldusage scenarios for chatbots. Zero-shot Prompting [55] in-
volves feeding tasks into the model without any examples
indicating the desired output, requiring the LLM to answer
the user’s questions without showing any examples in the
prompts. Few-shot Prompting [32] works by providing the
model with a small number of high-quality examples that
include the inputs and desired outputs of the target task. By
observingthesegoodexamples,themodelcanbetterunder-
stand human intent and the criteria for generating accurate
output.Chain-of-ThoughtPrompting[56]generatesaseries
of short sentences, known as chains of reasoning. These
sentencesdescribestep-by-stepreasoninglogicthatleadsto
a final answer, which can be more beneficial for tasks with
complexreasoningandlargermodels.RecursivePrompting
[57] is a problem solving methodology that involves break-
ingdowncomplexproblemsintosmaller,moremanageable
subproblemsandthensolvingthesesubproblemsrecursively
through a series of prompts.
2.5. Retrieval-augmented Generation
RAG [65] is an important technological breakthrough
in the field of natural language processing in recent years,
which effectively solves the inherent limitations of tradi-
tional LLM in terms of knowledge timeliness, factual accu-
racy, and interpretability by organically integrating the in-
formation retrieval system with the generative capability of
LLM.Thecoreinnovationofthetechnologyliesintheestab-
lishment of a dynamic knowledge injection mechanism that
enables the generative model to access and utilize external
knowledge bases in real time, thus significantly enhancing
the reliability and expertise of the generated content.
First Author et al.:Preprint submitted to ElsevierPage 4 of 24

LLM-based task planning for service robots: A review
1. Mov e to t he 
left side of the 
table .
2. Grasp the cup 
f r o m  t h e 
parallel
3. Pick up the cup 
gently .
4. Mov e to t he 
dining table .
5. Put down the 
cup gently .
Input
Robotics LLM -based Planner
instruction comprehension
environmental perception
task decomposition
subtask planning
Text
Audio
Image
Video
Agents consisting of Robotics and LLMs OutputAction Sequence
Figure 2:LLM-based task planning for service robots. The robot acquires information in the form of text, video, audio, etc.,
and passes the processed input signals to the LLM, which carries out the planning of detailed action sequences through natural
language processing, task decomposition, and so on.
RAGusuallyconsistsoftwomaincomponents,Retrieval
and Generation. First, the model retrieves a set of rele-
vantdocumentsfromapre-constructeddocumentcollection
based on an input query. The retrieved documents are then
fed into the LLM along with the query, and the model
generates the final answer based on these inputs. Some
typicalRAGapproachesincludeMulti-HeadRAG(MRAG)
[58], which utilizes multi-head attention to enhance mul-
tifaceted problem processing. Adaptive-RAG [59], which
dynamically selects strategies to cope with queries of vary-
ing complexity. Blended RAG [60], which fuses semantic
searchandhybridquerystrategiestoimproveaccuracy.Self-
RAG [61], enhancing model quality and factuality through
searchandself-reflection.AswellasIM-RAG[62],learning
introspective monologue integrates LLM and IR systems to
supportmulti-roundRAG.Theseapproacheshavetheirown
characteristics, and together they promote the development
of RAG technology.
2.6. Application of LLMs in Robotic Task
Planning
Inrecentyears,LLMshavemadesignificantprogressin
thefieldofnaturallanguageprocessing.Withthecontinuous
evolutionoftechnology,theapplicationofLLMshasbroken
throughthetraditionalscopeoftextgenerationandanalysis,
and begun to penetrate into emerging fields such as robot
taskplanning.AsshowninFig.2,thecombinationofLLMs
andhomeservicerobotsisdrivingthedevelopmentofmore
intelligent autonomous agents, with LLMs acting as the
“brain” of the service robots, which significantly improves
their ability in task planning.
Whentherobotreceivesambiguouscommandsfromthe
user,LLMisabletoparseandunderstandthemtoaccurately
capturetheuser’sintent[63,64,65].Forexample,inahome
service robot scenario, the user may say “put that thing on
thehighshelf",andLLMunderstandsthat“thatthing"refers
to the water cup on the table, and recognizes that “high"
refers to the top shelf of the bookshelf. and recognizes that“high up" refers to the top shelf of the bookcase. Based on
this, it combines logical reasoning to determine the optimal
execution plan, such as picking up the water cup first, then
planning a path to reach the bookshelf avoiding obstacles,
and finally placing the water cup at the specified location.
For better task planning, the robot may also rely on
VLMs to enhance environment perception [66, 67]. In the
above home service scenario, the VLM can recognize the
exact location of the water cup, the shape and height of the
bookshelf, as well as the obstacles (e.g., chairs or carpets)
on the path, providing precise physical information for task
planning. Faced with a complex task, the robot is able to
break it down into a series of executable sub-tasks, with
detailed, robot-executable task planning for each sub-task.
For example, a robot can break down “get a glass of water"
into steps such as “move to the table" and “grab the glass".
The steps of “navigating to the bookshelf", “adjusting the
posture",and“placingthecup"arebrokendownfor“placing
thecup".Inthisprocess,LLMsassistedindetailedplanning
to ensure that each subtask was properly executed, thus
improving the efficiency and accuracy of the overall task
planning.
3. Taxonomy
Against the backdrop of the intelligent development of
domesticservicerobots,taskplanningsystemsareevolving
from unimodal to multimodal fusion. Based on the differ-
encesininputmodalities,thispaperconstructsasystematic
classification framework, dividing LLM-driven task plan-
ning methods for domestic service robots into four ma-
jorcategories:Text-basedLLMPlanning,Vision-Language
Model Planning, Audio-based Planning, and Multimodal
Large Language Models Planning, as illustrated in Fig. 3.
This framework elucidates the technical characteristics and
performance boundaries under different modality combina-
tions. Here, we briefly summarize these four categories as
follows.
First Author et al.:Preprint submitted to ElsevierPage 5 of 24

LLM-based task planning for service robots: A review
Text
AudioVision
Multi -
modelLLM -based 
Service Robot 
Task PlanningLLM +PDDL
KnowNoLLM -Collab
ProgPrompt VLMapsVeriGraphReplanVLM
TaPA
CLAP
Wav 2CLIP
Beyond TextAudioPaLMLLMBindAVBLIP
RespLLM
mPnP -LLM
Figure 3:LLM-based task planning taxonomy for service
robots. From the point of view of different input modalities,
it is categorized into four categories such as text input, visual
input, audio input, and multimodal input.
Text-based LLM Task Planning.As the most fun-
damental implementation paradigm, this planning method
primarily relies on text-only input for task planning. Such
systemstypicallyemployTransformer-basedlanguagemod-
elsascoreprocessors,utilizingnaturallanguageunderstand-
ing (NLU) modules to parse user instructions and generate
correspondingactionsequences.However,duetothelackof
direct environmental perception capabilities, these systems
requirepre-constructedsemanticmapsorknowledgegraphs
to supplement environmental information.
Vision-based LLM Task Planning.This method sig-
nificantly enhances robots’ understanding of physical en-
vironments by integrating visual perception capabilities.
VLMs focus primarily on establishing semantic alignment
between visual and linguistic modalities. VLMs usually
adoptdual-encoderarchitecturesorcross-modalTransform-
ers to realize efficient visual-linguistic interactions, and
their typical applications include:VQA, image description
generation,visual localization. Through cross-modal atten-
tion mechanisms, the system establishes a joint "vision-
language" representation space, enabling complex instruc-
tions requiring spatial reasoning, such as "place the mug on
the dining table into the dishwasher".
Audio-based LLM Task Planning.Thesystemenables
the robot to understand and respond to voice commands by
introducinganaudioprocessingmodule.Suchsystemsusu-
allyincludespeechrecognitionmoduleandspeechsynthesis
module,whichcanconvertuser’svoicecommandsintotext
and generate corresponding action sequences. Through the
analysis of audio signals, robots can recognize and respond
to specific voice commands in noisy environments, thereby
improving their flexibility and adaptability in practical ap-
plications.Multimodal LLM Task Planning.The Multimodal
LLM (MLLM) cover not only vision and speech, but also
integrateaudio,sensordata,robotstate,andothermodalities
tosupportmorecomplextaskplanning.MLLMshavethree
main features: multimodal inputs, dynamic modal fusion,
and cross-modal reasoning. Such systems usually adopt a
hierarchical fusion strategy: modal alignment is achieved
through cross-attention at the feature level, and the con-
tribution weight of each mode is dynamically adjusted by
the gating mechanism at the decision level. In this way, the
MLLM Planning system can integrate multiple information
sources in a complex and changing home environment to
generate more accurate and flexible task planning.
4. Text-based LLM Task Planning
As the basic paradigm of LLM-driven task planning,
text-based systems establish a key baseline for human-
computer interaction through natural language processing.
These systems convert unstructured natural language in-
structions into executable action sequences by using the
semanticunderstandingabilityoftheconverterarchitecture,
representing the most computationally efficient method in
the modal spectrum.
The task planning system of LLM service robot based
on text faces two core challenges: one is how to effectively
deal with instructions of different complexity (from simple
short-term tasks to complex long-term planning), and the
other is how to make up for the lack of environmental
perception caused by pure text input. To this end, this
section will systematically elaborate the existing research
resultsfromtwodimensions:theplanningmethodbasedon
instruction complexity and the compensation technology of
environment-aware defects.
4.1. Planning method based on instruction
complexity
Due to the excellent performance of LLM in general-
ization and common sense reasoning, it can skillfully un-
derstand natural language instructions and perform logical
reasoning, translation and zero sample planning. This sec-
tionwillintroduceText-basedLLMPlanningintermsofthe
complexity of instructions, and some of the representative
work is shown in Table 2. In the face of simple short-term
tasks,LLMshowsexcellentplanningability.TheLLM-grop
proposed by Ding et al. [63] extracts common sense knowl-
edge about semantically valid object configuration from
LLM through the prompt function, and instantiates them
with tasks and motion planners in order to promote them to
differentscenegeometries,asshowninFigure4.LLM-grop
can achieve object rearrangement from natural language
commands to human alignment in various environments.
ProgPrompt, proposed by Singh et al. [64], is a program-
matic LLM prompt structure that utilizes the available op-
erations in the environment and the specification of similar
programsforobjects,sothattheplangenerationcanspanthe
environment,robotcapabilities,andtasks.Inaddition,LLM
First Author et al.:Preprint submitted to ElsevierPage 6 of 24

LLM-based task planning for service robots: A review
Table 2
Typical Literature on Text-based LLM Task Planning
Paper Year Core Innovation Limitations
Ding et al. [63] 2023Achieves cross-environment object rear-
rangement by combining semantic knowl-
edge with geometric planningRequires precise object labeling and limited
to tabletop-scale tasks
Singh et al. [64] 2023Enables generalized planning through pro-
grammatic prompts supporting multi-
modal specificationsDemands complex prompt engineering with
high computational overhead
Silver et al. [65] 2024Ensures plan validity through direct PDDL
integration with formal correctness guaran-
teesDepends on pre-defined PDDL domains
and scales poorly to novel objects
Ren et al. [68] 2023Provides statistical uncertainty quantifica-
tion while minimizing human interventionTends to produce conservative plans with
complex calibration requirements
Wei et al. [56] 2022Simplifies complex reasoning via chain-of-
thought prompting without model modifi-
cationLacks formal verification and sensitive to
prompt phrasing variations
Liu et al. [69] 2024Enables long-horizon planning through au-
toregressive goal decompositionSuffers from error accumulation in sub-
tasks requiring fine-tuning
Kannan et al. [70] 2024Facilitates multi-robot collaboration by
handling capability heterogeneityRequires centralized coordination introduc-
ing communication overhead
is often combined with the Planning Domain Definition
Language (PDDL) to improve the planning performance of
the LLM planner. Sliver et al. [65] directly provided the
planning problem in the PDDL syntax to LLM to generate
action sequences. Although the planning performance has
been improved, this method requires additional knowledge
to build a PDDL file. Liu et al. [71] skillfully used the
translationfunctionofLLMtoconvertnaturallanguageinto
PDDL problem, and then solved it by traditional planner,
and finally translated the solution back to natural language.
Similarly, Tomoya et al. [72] transform natural language
tasksintosymbolicsequencerepresentationsbyusingLLM,
and then derive the best task program by performing task
planning based on Monte Carlo Tree Search (MCTS).
In the realm of complex long-horizon planning, re-
searchers have explored several strategies to enhance the
capabilities of LLM-based robotic systems. One approach
involveshierarchicalanditerativeframeworks.Forinstance,
Chen et al. [73] proposed the Hierarchical Multiscale Dif-
fuser(HM-Diffuser),whichemploysahierarchicalstructure
totrainontrajectoriesextendedatmultipletemporalscales.
This is complemented by Progressive Trajectory Extension
(PTE), an augmentation method that iteratively generates
longer trajectories by stitching shorter ones, thereby ef-
ficiently managing tasks across different time horizons.
Another direction is the integration of explicit planning
modules. Erdogan et al. [74] introduced the Plan-and-Act
framework, which decouples the planning process into two
distinct components: a Planner model that generates struc-
tured,high-levelplanstoachieveusergoals,andanExecutormodel that translates these abstract plans into environment-
specific actions. To improve the quality of these plans,
their framework also incorporates a scalable synthetic data
generation method. Furthermore, a widely adopted strategy
to mitigate the complexity of long-horizon tasks is hier-
archical task decomposition. This approach breaks down
a complex mission into a sequence of manageable sub-
goals. For example, Wei et al. [56] leveraged chain-of-
thought (CoT) prompting to elicit step-by-step reasoning,
enablingthemodeltotacklecomplextasksmoreeffectively.
Building on this, Liu et al. [69] proposed DELTA, which
decomposes long-horizon objectives into an autoregressive
sequence of sub-goals for automated planners. Similarly,
Caoetal.[75]developedtheLLM-Collabframework,which
systematically divides complex planning into four stages-
analysis, planning, verification, and improvement-each fur-
ther decomposed into sub-tasks. Zhen et al. [76] merged
humanexpertisewithLLMsusingaspecialized“ThinkNet
Prompt” and a hierarchical decomposition strategy, while
Kannan et al. [70] employed LLMs to decompose tasks
for multi-robot collaboration, addressing the limitations of
single-function robots.
However, the possibility of hallucinations in the output
of LLM will accumulate due to the increase of planning
period. Therefore, in complex long-horizon task planning,
how to suppress or reduce the generation of hallucinations
is a major difficulty in using LLM for task planning. Ren et
al. [68] proposed KnowNo, a framework for measuring and
adjusting the uncertainty of LLM-based planners. KnowNo
is based on conformal prediction theory and provides sta-
tistical assurance for mission completion while minimizing
First Author et al.:Preprint submitted to ElsevierPage 7 of 24

LLM-based task planning for service robots: A review
LLM
TAMPGeometric Spatial 
RelationshipsSymbolic Spatial 
Relationships
Task Planner Motion PlannerService Request
“fork is on the left  
of bread plate ”
Action Sequence
Trajectory“Position” “width”
Task -motion plan
Figure 4:Schematic diagram of the architecture of LLM-grop.
Is there 
a plan ?
Task planner
Plan monitorPlan
LLMsKnowledge
AcquirerAction
Open worldGoal
SituationFeasible
YNempty not empty
Next
ActionAddReport No SolutionNot feasible Add “Action Precondition ”
Figure 5:Schematic diagram of the architecture of COWP in
paper.
manual help in complex multi-step planning settings. Park
et al. [77] designed a LLM uncertainty estimation method
to classify whether the command is certain or uncertain.
After determining the uncertainty, it classifies the input as
certain or uncertain according to a predefined threshold.
Once the command is classified as an uncertain command,
the LLM generation problem is used to interact with the
user to eliminate the ambiguity of the command. Ong et al.
[78] consider the uncertainty in planning by combining a
simple method, which emphasizes quantifying uncertainty
andexploringalternativepathsfortaskexecution.Bysetting
an appropriate probability threshold in skill selection, a
method of measuring uncertainty is established to select a
better path to perform tasks. Wang et al. [79] proposed a
task planning method based on constrained LLM prompt
scheme, which can generate executable action sequences
from commands. Furthermore, a special processing module
is proposed to deal with the LLM illusion problem, so as to
ensure that the results generated by LLM are acceptable in
the current environment.
4.2. Compensation technology of environmental
perception defects
Althoughthetext-basedLLMtaskplanningmethodhas
madegreatprogress,itstillhasaninherentlimitation,thatis,
the perception of the physical environment. The input form
of pure text makes researchers have to study compensationstrategies to deal with this challenge. The current research
mainly solves this problem through static knowledge injec-
tion.
Static knowledge injection is shown in LLM-grop [63].
Before the task planning of desktop object placement, the
current environmental information, including known items
suchasknives,forks,dinnerplates,andtheirexistingstates,
is pre-entered into LLM.Combined with the common sense
knowledge of LLM in the form of knowledge base, task
planning is performed instead of environmental awareness.
Mu et al. [80] introduced KGGPT, an innovative system
that integrates prior knowledge. This system operates by
extracting pertinent information from a knowledge graph,
transformingitintosemanticrepresentations,andthenlink-
ingthesetoChatGPT.KGGPTleveragesknowledgegraphs
toencoderoboticskills,taskregulations,andenvironmental
limitations. By doing so, it effectively bridges the gap be-
tween ChatGPT’s existing knowledge base and the practi-
cal requirements of real-world service environments. This
task planning assumes the invariance and stability of the
state of each object in the task space, but in a family en-
vironment where dynamic changes may occur, a prede-
fined knowledge base often results in inaccurate or unexe-
cutable task planning results. Dynamic state update better
handles the situation where the state of the object may
change in an open home environment. Ding et al. [81]
developedCOWP,anopen-worldrobotplanningsystemthat
ingeniously integrates a traditional knowledge-based task
planning framework with a pre-trained language model, as
shown in Figure 5. The latter serves as a powerful means
for acquiring domain-agnostic common-sense knowledge.
This synergy endows COWP with the unique ability to
apply universal common-sense knowledge to specialized
task-planning scenarios. As a result, COWP can effectively
handle unanticipated situations that robots may face during
the planning process, enhancing their adaptability in open-
ended and dynamic environments. However, this ability to
encounter and deal with emergencies is precisely what the
large language model of plain text input cannot achieve.
5. Vision-based LLM Task Planning
The inherent limitations of text-based LLM planners in
spatial perception and dynamic adaptation have promoted
the rise of VLMs and regarded them as a transformative
method in the field of robot task planning. By constructing
a unified visual-linguistic representation space, VLM effec-
tively bridges the perception-reasoning gap in a pure LLM
system [82]. In the context of service robot task planning,
the VLM-based method presents a diversified development
trend. In addition, the introduction of visual perception has
also stimulated researchers’ exploration in active task cog-
nition. In this section, we will explore the research progress
of VLM-based task planning methods and active task cog-
nition.
First Author et al.:Preprint submitted to ElsevierPage 8 of 24

LLM-based task planning for service robots: A review
Table 3
Typical Literature on Methodological and Frameworks Innovations
Paper Year Core Innovation Limitations
Li et al. [66] 2024Aligns visual attributes with LLM via VLM
for precise plans, enhanced by ontological
knowledgeRequires full attribute labeling and prede-
fined ontologies; struggles with novel ob-
jects
Tang et al. [83] 2025Automates 3D reconstruction from 2D in-
puts,enhancing3Dsceneunderstandingfor
dynamic environmentsAccuracydropsinsparseoroccludedscenes
and requires pre-trained VLM with 2D
grounding
Ni et al. [84] 2024Graph networks model object relations for
semanticreasoning,whileLLMenablesnat-
ural instruction decompositionGraph construction becomes costly in clut-
tered scenes and limited to pre-defined
relationship types
Ekpo et al. [85] 2024Scene graphs enable explicit spatial rela-
tionship verification and iterative LLM plan
correction ensures physical enforceabilityGraph construction latency and fails on
ambiguous spatial queries
Mei et al. [86] 2024Dual-loop correction improves error recov-
ery robustness and maintains plan consis-
tency during dynamic environment changesIterative corrections increase task comple-
tion time and dependent on VLM’s visual
grounding accuracy
Wu et al. [67] 2023Tight LLM-VLM alignment ensures physi-
cally feasible plans and context-aware con-
straint handlingRequires complete 3D object models for
constraint verification and struggles with
implicit physical rules
5.1. Methodological and Frameworks Innovations
Integrating VLMs into the field of service robot task
planning is a major improvement on the traditional text-
based method. This part deeply studies the methods and
frameworksofusingVLMstoenhancethespatialperception
anddynamicadaptabilityofservicerobots.Bybridgingthe
gap between visual perception and language understanding,
VLMs enable robots to interpret complex environmental
cuesandperformtaskswithhigheraccuracyandflexibility.
This section will introduce Text-based LLM Planning in
terms of the complexity of instructions, and some of the
representative work is shown in Table 3.
In spatial representation and semantic comprehension,
Huang et al. [87] proposed VLMaps, which is a spatial
mapping representation that directly combines pre-trained
visuallanguagefeatureswith3Dreconstructionofthephys-
ical world. In this way, service robots can position space
targets outside object-centric space targets during mission
planning and execution, such as ‘between the TV and the
couch’. At the same time, Tang et al. [83] proposed a
new framework that uses a 2D prompt synthesis module
to enable VLM to train on 2D images and texts, and can
independentlyextractaccurate3Dspatialinformationwith-
outmanualintervention,therebysignificantlyenhancing3D
scene understanding, enabling robots to plan and perform
actions adaptively in a dynamic environment.
For task decomposition and planning refinement, Ni et
al. [84] proposed a new method called graph-based robot
instruction decomposer (GRID), as shown in Figure 6. This
method uses the LLM and graph attention networks to
encode the object attributes and relationships in the graph,
so that the robot can obtain the semantic knowledge widely
Robot Graphrobotbook -
shelf
study 
roombeside
ininScene Graphfloorstudy 
room
kitchenbook -
shelf
onin
onInstruction“please take the 
book from bookshelf 
to  table”
LLM 
Encoder
Feature
EnhancerTask
Decoder<Action >
pick
<Object >
bookFigure 6:The architecture diagram of GRID.
observed in the environment from the scene graph. Ekpo et
al. [85] proposed VeriGraph, which is a new framework for
integratedVLMforrobotplanning.VeriGraphusesscenario
graphsasintermediaterepresentationstocapturekeyobjects
and spatial relationships, thereby improving planning vali-
dation and optimization. The system generates a scene map
fromtheinputimageandusesittoiterativelycheckandcor-
rect the sequence generated by the LLM-based task planner
to ensure compliance with constraints and enforceability.
Regarding dynamic adaptation and self-correction, Mei
etal.[86]proposedaReplanVLMframeworkforrobottask
planning,asshowninFigure7.Aninternalerrorcorrection
mechanism and an external error correction mechanism are
proposed for error correction at the corresponding stage.
Shirai et al. [88] proposed a visual language interpreter
ViLaln, which is a new problem description framework
generated by the most advanced LLM and visual language
model. It can optimize the generated problem description
through the error message feedback of the symbol planner.
First Author et al.:Preprint submitted to ElsevierPage 9 of 24

LLM-based task planning for service robots: A review
          Give me a 
           green cube
Prompt :
You are a programmer ..
image
Task Planning Code Generation
Decision
VLM
Extra
InnerN
YY
N
End
 Environment
Figure 7:The framework of ReplanVLM.
Inembodiedtaskexecutionandphysicalconstraintman-
agement, Wu et al. [67] developed a TaPA for landing
planningwithphysicalsceneconstraintsinembodiedtasks.
Theagentgeneratesanexecutableplanbasedontheobjects
in the scene by aligning the LLM with the visual language
perception model. Li et al. [66] proposed the fine-grained
task planning FGTP, which aligns the object attributes with
their corresponding images through VLM to obtain the
actualattributesoftheobject.Thisisamethodthatcombines
object ontology knowledge with the LLM to create action
sequences. Inordertodealwiththedifficultiesencountered
in dealing with complex tasks and interacting with the
environment effectively, Luan et al. [89] introduced VLM
as an environment-aware sensor and assimilated the results
into LLM, thus integrating task objectives with environ-
mentalinformationandenhancingtheenforceabilityoftask
planning output. Huang et al. [90] used the combination
of VLM and LLM to perform the task of robot-to-human
handover with semantic object knowledge, which enhanced
the perception and interaction of robots in a dynamic en-
vironment and paved the way for more seamless human-
machine collaboration.
Furthermore, Zhang et al. [91] developed GPTArm and
proposed a robot task processing framework RTPF, which
integrates real-time visual perception, situational reasoning
and autonomous strategy planning, so that the manipulator
can interpret natural language commands, decompose user-
defined tasks into executable sub-tasks, and dynamically
recover from errors. Wake et al. [92] proposed a ready-to-
use multi-mode task planner using VLM and LLM, and
introducedapipelinetoenhancethegeneralvisuallanguage
model GPT-4V to promote the one-time visual teaching of
robots, successfully transforming human actions in videos
into robot executable programs. Liu et al. [93] proposed
a VLM-driven method to understand scenes in unknown
environments. The visual language model is used as the
skeleton of the language model for image description and
scene understanding, which further proves that the taskplanningabilityoftheservicerobotinthehomeenvironment
issignificantlyimprovedthroughthevisuallanguagemodel.
5.2. Active Task Cognition
Active task cognition necessitates that robots possess
the capability to comprehend their environment and au-
tonomously deduce task-related information. Fundamen-
tally, active task cognition can be characterized as a visual
reasoning task, wherein the reasoning process serves as
the cornerstone of the cognitive procedure. This form of
visual reasoning extends beyond mere object identification
withinascene[94];itencompassesanunderstandingofthe
interactions among these objects, as well as their respective
significance and function in the real world. This section
elaborates on active task cognition from visual reasoning,
which is typical as shown in Table 4.
In the realm of visual cognition research, an emerging
body of literature underscores the criticality of representing
relationalstructuresandsemanticinformationgleanedfrom
images. Mo et al. [95] introduced a graph learning-based
multimodal prior guided segmentation framework, which
innovatively addresses the dual challenges of feature ex-
traction and cross-modal correspondence. This framework
enables the efficient extraction of modality-specific features
while establishing robust regional correspondences across
multiple modalities, thereby facilitating more comprehen-
sive visual understanding. Yang et al. [101] proposed an
alternativeapproachcenteredonemotionalreasoning.Their
methodology involves constructing an emotional graph that
integratessemanticconceptsandvisualfeatures.Leveraging
Graph Neural Network (GCN) techniques, they perform
reasoningoperationsontheemotionalgraph,ultimatelygen-
erating emotion-enhanced object features. These advance-
ments collectively highlight the potential of graph-based
models in enriching visual cognition by encoding complex
semantic and relational information.
In addition, integrating the scene graph into the intelli-
gentsystemnotonlyenhancesthevisualcognition,butalso
emphasizestherelationshipandattributesofasingleobject,
which promotes the scene understanding [102, 103, 104].
Liu et al. [96] devised a region-aware attention learning
approach. This method emphasizes fine-grained visual re-
gions over coarse-grained bounding box features, thus op-
timizing the image scene graph generation process. Mean-
while, Zhang et al. [97] innovatively incorporated cooking
logicintofoodimagecomponentrecognition.Byextracting
semantic information from images, their method directly
generatescorrespondingrecipesandpreparationtechniques,
bridgingvisualunderstandingandpracticalapplication.Jiao
etal.[105]plannedanddesigneda3Dscenegraphrepresen-
tationfortherobotsequencebyabstractingthescenelayout
of robot-scene interaction. These models can effectively
capture the spatial and semantic relationships of objects to
promote the robot’s cognition of tasks.
In addition, the application of Visual Question Answer-
ing (VQA) technology also contributes to enhancing the
First Author et al.:Preprint submitted to ElsevierPage 10 of 24

LLM-based task planning for service robots: A review
Table 4
Typical Literature on Visual Reasoning
Paper Year Core Innovation Limitations
Mo et al. [95] 2021Proposed a graph-based multimodal frame-
work for joint learning and segmentationRequires paired multimodal data; sensitive
to modality misalignment
Liu et al. [96] 2021Replaces bounding-box features with fine-
grained region attention; more precise rela-
tionship prediction in scene graphsHigher computational cost than bounding-
box approaches and needs detailed region-
level annotations for training
Zhang et al. [97] 2022Novel 3D scene graph structure combining
geometric and semantic informationReal-time graph updates may be challeng-
ing in dynamic environments
Xie et al. [98] 2022Integrates visual concepts with external
knowledge bases; generates semantically
rich questions about imagesRelies on manually curated knowledge
bases; limited to domains covered by the
knowledge base
Hao et al. [99] 2024Chain-of-Thought enhanced spatial reason-
ing; improves SQA accuracy for complex
spatial queriesDependent on LLM reasoning reliability;
limited to predefined spatial primitives
Cheng et al. [100] 2024Specialized in geometric relationship rea-
soning; bridges visual perception with sym-
bolic reasoningProne to LLM hallucination in spatial de-
scriptions; requires accurate object detec-
tion as input
extraction of semantic information. Frankli et al. [106] pro-
posed RobotVQA, an innovative robot vision architecture
designed for VQA tasks. The architecture uses RGB or
RGBD images of the scene where the robot is located as
the original input, and accurately identifies related objects
in the scene through advanced object detection algorithms.
Onthisbasis,thesemanticmapofthesceneissystematically
constructedbyanalyzingthequalitativespatialrelationship,
so as to provide structured environmental information sup-
port for subsequent intelligent decision-making and task
execution.Xieetal.[98]proposedaknowledge-basedVQA
model, which innovatively integrates visual concepts and
non-visual knowledge to generate deep semantic problems
for images, and significantly improves the scene cognition
level of robots. Similarly, Hao et al. [99] proposed a novel
spatial reasoning paradigm, which significantly improves
the performance of Situational Question Answering (SQA)
system in spatial relationship reasoning and complex query
processingbyorganicallyintegratingbasicmodelandthink-
ing chain mechanism. This paradigm enables the system
to analyze spatial semantic information more accurately
through a structured reasoning process. In order to help
robots understand scene context information, Wang et al.
[107] constructed a chain guided learning model for graph-
ical question answering (DQA). The model relies on the
LLM to guide the graph analysis tool, which significantly
improves the accuracy of graph analysis and enriches the
backgroundknowledgesystem.Coincidentally,Chengetal.
[100] proposed spatial region GPT (SpatialRGPT), which
is dedicated to spatial perceptual VQA. By strengthening
spatial perception and reasoning ability, it can effectively
promote the cognitive process of robot tasks.In the above research, active task cognition mainly fo-
cuses on three technical directions: Graph-based Reasoning
Systems: Several studies [95, 96, 97, 105] have developed
sophisticated graph representations that enable robots to
activelyreasonaboutobjectrelationshipsandtaskcontexts.
For instance, Jiao et al. [105] proposed a 3D scene graph
representation that abstracts scene layouts and robot-scene
interactions,allowingrobotstodynamicallyupdatetheirun-
derstandingoftheenvironmentduringtaskexecution.VQA
Frameworks: Advanced VQA systems [98, 99, 106, 107]
empowerrobotstoactivelyquerytheirvisualunderstanding.
The RobotVQA system [106] demonstrates how robots can
construct semantic maps by analyzing qualitative spatial
relationships through a question-answering paradigm. Mul-
timodalChain-of-ThoughtReasoning:Newframeworkslike
SpatialRGPT [100] combine visual perception with step-
by-step reasoning, enabling robots to actively verify their
understanding through multimodal evidence. Despite these
advances,currentsystemsstillfacelimitationsingenuineac-
tive cognition. Most approaches remain reactive rather than
proactive, requiring explicit human queries or predefined
triggers rather than autonomously identifying task-relevant
information.Atpresent,howtoconstructareasoningmodel
with active task cognitive ability based on VLM and visual
perception information has become a key challenge to be
overcome.
6. Audio-based LLM Task Planning
Althoughvisualandtextualmodalitiesdominateinrobot
taskplanning,audiosignalsprovideauniquechannelforthe
perceptionofservicerobots,especiallyinvisualdeprivation
scenarios (e.g., dark environments or assisted care). Audio-
based LLM mission planning leverages voice commands,
First Author et al.:Preprint submitted to ElsevierPage 11 of 24

LLM-based task planning for service robots: A review
2. Use pretrained encoders for zero -shot predictionAudio 
Encoder
Text 
EncoderBird song
Thunder rumbling
Knocking on the door...
ClassesTesting
1 1AT
...
1 2AT
1 3AT
1 NAT
Bird song
1 1AT
...
1 2AT
1 3AT
1 NAT
2 1AT
...
2 2AT
2 3AT
2 NAT
3 1AT
...
3 2AT
3 3AT
3 NAT
...
N 1AT
...
N 2AT
N 3AT
N NAT
...
...
...
...Text 
EncoderAudio 
Encoder
1. Contrastive Pretraining
Figure 8:The architecture diagram of CLAP.
 
Figure 9:Workflow of the proposed AudioCLIP model.
 According to <audio > ,
you are allowed to use or
partially use the
information about the
following events present
in the audio :
[Shout , Yell, Giggle , 
Sigh]
(Audio Tags )AST
Cross Attention
Self Attention
FFN
Cross Attention
Multi -layer
 Aggregator...x N
    Audio QFormerQueries
      Soft Prompt
MLP MLP MLP
Large language ModelFrozenFinetune
❄🔥
Text Instruction 
Prompt
Response❄
🔥🔥 🔥
LoRA   o   🔥
Figure 10:The schematic diagram of GAMA.
ambient sound, and acoustic cues to enable robots to in-
terpret implicit user intents and dynamic contexts. How-
ever,comparedwiththevigorousdevelopmentofLLMtask
planning based on text and visual input, audio-driven task
planning still faces many challenges, and related research
is in a stage of rapid development.This part studies how
LLMs handle audio input for task planning, and elaborates
on the work of Audio Language Model (ALM), acoustic
scene perception and speech interaction. Typical examples
are shown in Table 5.
In terms of the basic model architecture, several break-
through studies have laid a solid foundation for the joint
representationofaudioandlanguage.AsshowninFigure8,
theCLAP(ContrastiveLanguage-AudioPretraining)model
proposed by Elizalde et al. [108] effectively connects the
languageandaudiomodalitiesthroughadual-encoderstruc-
tureandacontrastivelearningobjective,mappingaudioand
text descriptions into a joint multimodal space to enable
flexible class prediction during inference and significantly
enhancecross-modalretrievalperformance.Buildingonthis
work, the team led by Guzhov [109] integrated the high-
performance audio model ESResNeXt with the text-imagecontrastive model CLIP, as shown in Figure 9, constructing
a three-modal hybrid architecture that excels in environ-
mental sound classification tasks and extends the zero-shot
learning capability of the basic model to the audio domain.
Furthermore, Wu et al. [115] proposed Wav2CLIP, which
innovativelyprojectsaudio,images,andtextintotheshared
embedding space of CLIP, offering an efficient solution for
tasks like zero-shot sound classification and demonstrat-
ing strong adaptability and generalization in multimodal
scenarios, thus advancing the field of audio-language joint
representation.
To bolster models’ comprehension of intricate audio
scenes, researchers have engineered diverse specialized ar-
chitectures. The Audio Flamingo model, devised by Kong
et al. [110], exhibits remarkable few-shot learning and con-
versational prowess, enabling it to rapidly adapt to unseen
task scenarios. Gong et al. [116] introduced LTU (Listen,
Think, and Understand), a novel audio foundation model
that demonstrates robust performance and generalization
on traditional audio tasks, including classification and cap-
tioning.Notably,LTUshowcasesemergentaudioreasoning
and understanding capabilities hitherto absent in existing
models.AsshowninFigure10,Ghoshetal.[117]proposed
GAMA, a general-purpose large audio language model that
integratesadvancedaudiocomprehensionandsophisticated
reasoningabilities,therebyofferinganovelparadigmforthe
semantic analysis of ambient sounds.
In the realm of voice interaction, numerous studies
have focused on enhancing robots’ ability to understand
and execute spoken instructions. To handle the subtleties
of oral commands in scenarios like social navigation, Sun
et al. [118] proposed “Beyond Text”, a method that im-
proves LLM decision-making by integrating audio tran-
scription with functional sub-parts, seamlessly combining
text-based guidance with a human-auditory information
language model and marking a significant advancement in
socialrobotnavigationandbroaderhuman-robotinteraction.
Meanwhile, Chen et al. [111] introduced a novel Speech
Augmented Language Model (SALM), featuring a frozen
text LLM, an audio encoder, a modality adapter module,
and LoRA layers to adapt to voice input and task instruc-
tions. The unified SALM not only achieves performance
comparabletotheConformerbaselineforautomaticspeech
recognition (ASR) and speech translation (AST) tasks but
also demonstrates zero-shot context learning capabilities,
highlighting its adaptability in voice interaction scenarios.
Moreover, Rubenstein et al. [112] introduced AudioPaLM,
a sophisticated large language model designed for speech
understanding and generation. AudioPaLM integrates the
text-based language model PaLM-2 [119] and the speech-
based language model AudioLM [120] into a unified mul-
timodal architecture. This innovative integration endows
AudioPaLM with the ability to process and generate both
text and speech, enabling it to perform tasks such as speech
recognition and speech-to-speech translation with remark-
able efficiency and accuracy.
First Author et al.:Preprint submitted to ElsevierPage 12 of 24

LLM-based task planning for service robots: A review
Table 5
Typical Literature on Audio-based LLM Task Planning
Paper Year Core Innovation Limitations
Elizalde et al. [108] 2023Dual-encoder contrastive learning for
audio-text alignment; enables zero-shot
audio classification and retrievalRequires large-scale aligned audio-text
pairs; struggles with fine-grained audio se-
mantics
Guzhov et al. [109] 2022Tri-modal fusion via CLIP extension; ES-
ResNeXt audio encoder for environmental
sound classificationLimited to coarse sound categories; no joint
training of all modalities
Kong et al. [110] 2024Few-shot learning for unseen audio tasks;
conversational audio reasoning via LLM
integrationContext window limits long audio process-
ing; overfits to frequent sound patterns
Chen et al. [111] 2024Maintains LLM knowledge while adding
audio via adapters; Zero-shot adaptation to
new voice instructionsAudio encoder limits real-time processing;
degrades with noisy speech
Rubenstein et al. [112] 2023Unified PaLM-2 + AudioLM architecture;
bidirectional speech-text generationRelies on high-quality parallel speech-text
corpora; generated speech lacks emotional
nuance
Wang et al. [113] 2024Gradient-free adaptation to speakers; in-
context learning with few audio-text exam-
plesRequires representative few-shot examples;
limited to vocabulary seen in context
Deshmukh et al. [114] 2023Unified text-generation framework for all
audio tasks; eliminates task-specific fine-
tuningNo explicit audio-text alignment; struggles
with novel sound events
Toassesstheperformanceofaudiocomprehensionmod-
elsintasksdemandingexpert-levelknowledgeandintricate
reasoning, Sakshi et al. [121] introduced MMAU (Mul-
timodal Multilingual Audio Understanding), a benchmark
featuring 10,000 meticulously curated samples along with
manuallyannotatednaturallanguagequestionsandanswers,
covering speech, environmental sounds, and music. In a
different yet equally impactful vein, Wang et al. [113] pro-
posed a novel Speech-based In-context Learning (SICL)
approach that harnesses a few context examples—paired
spoken words and labels—from specific dialects or speak-
ers to achieve language- or speaker-level model adaptation
duringtestingwithoutgradientdescent,effectivelyresolving
the adaptability issues stemming from dialect and speaker
disparities. At the application level, Salewski et al. [122]
developedZerAuCap,amodelguidedbyapre-trainedaudio
languagemodeltogeneratecaptionsthataccuratelydescribe
audio content. It utilizes audio context keywords to prompt
the language model, enabling the generation of text closely
associated with the sounds. Meanwhile, addressing the lim-
itation in generating appropriate language for open-ended
tasks like audio captioning and audio Q&A, Deshmukh et
al.[114]introducedPengi.Thisinnovativemodelcapitalizes
on transfer learning by formulating all audio tasks as text
generationtasks.Pengi’sunifiedarchitecturecanseamlessly
handle both open and closed tasks, eliminating the need
foradditionalfine-tuningortask-specificmodifications,thus
offering a versatile and efficient solution for various audio-
language processing needs.Despite these advances, audio LLM planning still faces
many challenges. The ClozeGER error correction frame-
work proposed by Hu et al. [123] for speech recognition
errors improves the fidelity of the corrected output by in-
troducing a multimodal LLM (SpeechGPT) to receive the
sourcespeechasadditionalinput,butitsrobustnessincom-
plex noise environments still needs to be improved.
7. Multimodal-based LLM Task Planning
With the rapid development of LLM, limited by its nar-
row sensory input and fragile environmental understanding
ability, people are no longer satisfied with a single-modal
LLM, but try to integrate the input of multiple modali-
ties [124] and integrate visual, language, audio and other
signals into a MLLM. As a transformative paradigm of
robot planning, MLLM enables robots to understand fuzzy
instructions,adapttodynamicenvironments,andlearnfrom
cross-modal correlations. In this section, we will introduce
thecurrentresearchprogressandapplicationofmultimodal
large language models.
7.1. Multimodal Large Language Models
In this section, we will explore a series of typical
MLLMs and analyze the differences between them. The
FlamingoarchitectureproposedbyAlayracetal.[125]firstly
usedthequery-basedcross-attentionmechanism,thatis,the
‘sensor resampler’. As shown in Figure 11, the architecture
connects the powerful pre-trained pure visual model with
First Author et al.:Preprint submitted to ElsevierPage 13 of 24

LLM-based task planning for service robots: A review
Vision 
Encoder
Perceiver
Resampler
Interleaved visual /text data
This is a very cute dog .
This is  <image > This is a 
very cute dog . 
<image > This isProcessed text n-th GATED XATTN -DENSEn-th LM block
 1st GATED XATTN -DENSE 1st LM block...Output : text
 a very angry cat . Pretrained and frozen  Trained from scratch
Figure 11:The architecture diagram of Flamingo.
BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models
ImageEncoder
Input Image
LearnedQueriesFeed ForwardCross AttentionSelf Attentionx Nx NImage-Text Contrastive LearningFeed ForwardSelf Attentiona cat wearing sunglassesInput TextImage-Grounded Text GenerationImage-TextMatchingfor every other blockAttention MaskingUni-modalSelf-Attention MaskQTQTMulti-modal CausalSelf-Attention MaskQTQTBi-directionalSelf-Attention MaskQTQTImage-Text MatchingImage-Text Contrastive LearningImage-Grounded Text GenerationQ: query token positions;  
…unmaskedmaskedT: text token positions.Q-Former
bidirectionaluni-modalmutlimodal causalxx
Figure 2. (Left) Model architecture of Q-Former and BLIP-2’s first-stage vision-language representation learning objectives. We jointly
optimize three objectives which enforce the queries (a set of learnable embeddings) to extract visual representation most relevant to the
text. ( Right ) The self-attention masking strategy for each objective to control query-text interaction.
ules that share the same self-attention layers: (1) an image
transformer that interacts with the frozen image encoder
for visual feature extraction, (2) a text transformer that can
function as both a text encoder and a text decoder. We create
a set number of learnable query embeddings as input to the
image transformer. The queries interact with each other
through self-attention layers, and interact with frozen image
features through cross-attention layers (inserted every other
transformer block). The queries can additionally interact
with the text through the same self-attention layers. Depend-
ing on the pre-training task, we apply different self-attention
masks to control query-text interaction. We initialize Q-
Former with the pre-trained weights of BERT base(Devlin
et al., 2019), whereas the cross-attention layers are randomly
initialized. In total, Q-Former contains 188M parameters.
Note that the queries are considered as model parameters.
In our experiments, we use 32 queries where each query
has a dimension of 768 (same as the hidden dimension
of the Q-Former). We use Zto denote the output query
representation. The size of Z(32×768) is much smaller
than the size of frozen image features ( e.g.257×1024 for
ViT-L/14). This bottleneck architecture works together with
our pre-training objectives into forcing the queries to extract
visual information that is most relevant to the text.
3.2. Bootstrap Vision-Language Representation
Learning from a Frozen Image Encoder
In the representation learning stage, we connect Q-Former to
a frozen image encoder and perform pre-training using
image-text pairs. We aim to train the Q-Former such that the
queries can learn to extract visual representation that is most
informative of the text. Inspired by BLIP (Li et al., 2022),
we jointly optimize three pre-training objectives that share
the same input format and model parameters. Each objec-
tive employs a different attention masking strategy between
queries and text to control their interaction (see Figure 2).
Image-Text Contrastive Learning (ITC) learns to align
image representation and text representation such that their
mutual information is maximized. It achieves so by contrast-
ing the image-text similarity of a positive pair against thoseof negative pairs. We align the output query representation
Zfrom the image transformer with the text representation
tfrom the text transformer, where tis the output embed-
ding of the [CLS] token. Since Zcontains multiple output
embeddings (one from each query), we first compute the
pairwise similarity between each query output and t, and
then select the highest one as the image-text similarity. To
avoid information leak, we employ a unimodal self-attention
mask, where the queries and text are not allowed to see each
other. Due to the use of a frozen image encoder, we can
fit more samples per GPU compared to end-to-end meth-
ods. Therefore, we use in-batch negatives instead of the
momentum queue in BLIP.
Image-grounded Text Generation (ITG) loss trains the
Q-Former to generate texts, given input images as the con-
dition. Since the architecture of Q-Former does not allow
direct interactions between the frozen image encoder and
the text tokens, the information required for generating the
text must be first extracted by the queries, and then passed
to the text tokens via self-attention layers. Therefore, the
queries are forced to extract visual features that capture all
the information about the text. We employ a multimodal
causal self-attention mask to control query-text interaction,
similar to the one used in UniLM (Dong et al., 2019). The
queries can attend to each other but not the text tokens. Each
text token can attend to all queries and its previous text to-
kens. We also replace the [CLS] token with a new [DEC]
token as the first text token to signal the decoding task.
Image-Text Matching (ITM) aims to learn fine-grained
alignment between image and text representation. It is a
binary classification task where the model is asked to pre-
dict whether an image-text pair is positive (matched) or
negative (unmatched). We use a bi-directional self-attention
mask where all queries and texts can attend to each other.
The output query embeddings Zthus capture multimodal
information. We feed each output query embedding into a
two-class linear classifier to obtain a logit, and average the
logits across all queries as the output matching score. We
adopt the hard negative mining strategy from Li et al. (2021;
2022) to create informative negative pairs.
3
Figure 12:TheschematicdiagramofBLIP-2.Reproducedwith
permission from [126]. Copyright 2023, JMLR.org.
the pure language model, and can handle any alternating
visual and text data sequences, as shown in Figure 12. This
innovative way to build a strong visual language interaction
module,showingtheexcellentabilitytolearninthecontext
of the environment. Similarly, BLIP-2 [126] uses a similar
method in the image coding process, using the Qformer
model to extract image features. Subsequently, the model
promotestheinteractionbetweenimageandtextthroughthe
cross-attention mechanism. When fine-tuning for a specific
downstream task data set, BLIP-2 will unlock the visual
encoder and fine-tune it together with Qformer.
In addition, Yang et al. [127] proposed MM-REACT,
which combines ChatGPT with various visual models to
complete multimodal tasks. The system is similar to the
previousresearchresultsusingthevisualquestionanswering
captionmodelandthelanguageimagemodel.However,the
special feature of MM-REACT is that it has the ability to
independently determine whether to call the visual model.
TheLLaMA-AdapterproposedbyZhangetal.[128]realizes
an efficient fine-tuning process through an adapter, thus
expandingtomultimodalapplicationscenarios.MiniGPT-4
proposedbyZhuetal.[129],combinesBLIP-2andVicuna,
andusesaprojectionlayertoalignthefrozenvisualencoder
with the frozen high-level large language model Vicuna,
thereby achieving advanced multimodal functions. LLaVA
[130] uses GPT-4 to generate command fine-tuning data,
and further strengthens the command tracking ability of the
multimodallargelanguagemodelbytrainingthevisualcom-
mandadjustmentdata.LLaVA-1.5[131]isextendedonthis
basis,andtheVQAdatasetisintegratedintotheinstruction
adjustment data, which greatly improves the performance
in a series of benchmark tests. Kosmos-2, introduced by
Pengetal.[132],implementsnewcapabilitiesforperceivingobjectdescriptions(i.e.,boundingboxes)andanchoringtext
to visual scenes by representing reference expressions as
links in the Markdown format, where objects are described
as sequence of position markers.
In contrast, QWen2-VL [133] adopts a more complex
three-stagetrainingprocessandintroducesaNaiveDynamic
Resolution mechanism, so that the model can dynamically
processimageswithdifferentresolutionsintodifferentnum-
bers of visual markers. This method enables the model to
generate more efficient and accurate visual representation,
which is closely related to the human perception process.
ThemPLUG-OwlintroducedbyYeetal.[134],throughthe
modular learning of the basic large language model, com-
bined with the visual knowledge module and the visual ab-
stractormodule,givesthelargelanguagemodelmultimodal
capabilities. This method makes full use of the synergy
between modalities and improves the performance of plain
texttasksandmultimodaltasks.Atthesametime,Otterpro-
posedbyLietal.[135]improvedtheOpenFlamingomodel,
which focuses on improving the execution of instructions
andeffectivelyusescontextsamples,showingextraordinary
proficiency in multimodal perception, reasoning and situ-
ational learning. CogCoM, introduced by Qi et al. [136],
introduces ‘Chain of Manipulations’, which is a mechanism
that allows visual language models to solve problems step
by step with evidence. After training, the model can solve
various visual problems by actively triggering internal op-
erations(suchaspositioning,magnification)andgenerating
results (such as bounding boxes, images), without the need
for external tools, while allowing users to trace the causes
of errors. Yan et al. [137] introduced a new framework,
Visual Grounding Through Fine-Grained Reward Model-
ing (ViGoR), which uses fine-grained reward modeling to
significantly enhance the visual positioning ability of large-
language visual models based on pre-training baselines.
Different modal data have significant differences in
terms of presentation, resolution and semantic granularity,
and how to unify them to extract common information
while retaining their unique details is the core challenge
in the fusion process. The above studies have developed a
variety of fusion techniques to try to solve these problems,
mainly including attention mechanisms, feature alignment,
gating mechanisms and graph neural networks. Attention
mechanisms have emerged as the predominant fusion strat-
egy in Transformer-based architectures. Cross-attention en-
ables dynamic association between modalities [125], while
self-attention captures intra-modal relationships. This dual
mechanism allows flexible information interaction tailored
to specific task requirements. Feature alignment techniques
create shared latent spaces for heterogeneous data. Con-
trastive learning methods like CLIP [138] align modalities
bymaximizingsimilarityofmatchedpairswhileminimizing
unrelated ones. The Q-Former module in BLIP-2 [126]
extends this through shared query vectors that learn unified
cross-modal representations. Gating mechanisms provide
dynamic information flow control. Lightweight adapters, as
seeninLLaMA-Adapter[128],enablecontext-awarefeature
First Author et al.:Preprint submitted to ElsevierPage 14 of 24

LLM-based task planning for service robots: A review
2.2 Efficient Insertion of External Modalities
After having connected encoders to LLM’s transformer blocks, there are multiple choices of insert-
ing multimodal information from encoders to LLM. Insertion into the LLM’s latent space enables
more efficient cross-modal interaction [47, 51] than only into the input layer, due to more oppor-
tunities to interact with different levels of feature representations. Early work adopts FiLM-like
weighting schemes [15, 48, 29, 9] to condition one modality on others in intermediate layers, but
only hard-coded two modalities and require domain-specific structure designs from scratch. Later
schemes used cross-MHA mechanisms for cross-modal interaction [55, 56, 36, 64], but cannot be
applied on decoder-only LLMs. To avoid changing the pre-trained LLM structures, recent work
[51] directly concatenates the projected multimodal tokens with output sequences from multiple
LLM blocks. However, LLM blocks for such concatenation are arbitrarily selected and not opti-
mized for training accuracy or speed. Extending the output sequence length for one LLM block will
also affect the token sequence length in other blocks and reduce accuracy.
These limitations of existing schemes motivate us to design a more flexible and generic method
to insert multimodal information into pre-trained LLMs, and our basic idea is to allow adaptive
weighting of multimodal tokens being inserted into the K-V set in LLM blocks. In this way, we
can establish optimal connections between encoders and LLM at runtime, for efficient cross-modal
interaction.
2.3 FLOPs Model for Modality Adaptation
Layer 1
Layer 2
Layer 3
Layer 4Input Pred .
Labeldy4 dy3 dy2
dw1 dw2 dw3 dw4y1 y2 y3 y4
forward
backward
Figure 4: Backpropagation of a 4-layer dense NNOur approach is inspired by the backpropa-
gation model for modality adaptation. Typ-
ically, the FLOPs of backpropagation can
be decomposed into two parts using the
chain rule. As shown in Figure 4, when
training a 4-layer dense neural network,
each layer computes i)the activation gra-
dient dyiand passes it backward, and ii)
computes the weight update dwiusing
dyi+1from the upstream layer. Freezing some layers can eliminate the FLOPs of computing these
layers’ weight updates, but activation gradients will still be computed. For example, when freezing
layer 2 to 4, the activation gradients from dy2tody4still need to be computed for layer 1 to compute
its weight update. Due to the generality of the chain rule, this mechanism applies to any other types
of models, including transformer blocks in LLMs.
Existing work inserts new modalities into LLM’s input layer through trainable projectors, and the
inserted projectors can be considered as the model’s first layer. Even when all the LLM layers are
frozen, the projector still needs activation gradients to be passed through the entire LLM. Instead,
our design connects input modalities into the last layers of LLM, to minimize the training cost at
runtime.
RGB 
Encoder
LiDAR 
Encoder
Key 
Aligner
Value 
AlignerLiDAR 
keys
LiDAR 
values
Key 
Aligner
Value 
AlignerRGB 
keys
RGB 
valuesMultimodal 
keys (K)
Multimodal 
values (V)Modality 2: LiDAR Point Cloud
Modality 1: RGB Camera View
LLM Block
LLM BlockLLM Block
MHA
K V Q
Q: Are there any pedestrians 
at the right sidewalk?A: Yes
LLM Decoder
Concat
×𝛼𝛼1×𝛼𝛼2×𝛼𝛼𝑁𝑁Trainable latent 
connection
Environment 
Changes
day  night
mountUnimodal Encoders
The last N 
LLM blocks
: the model components to be retrained at runtime
Figure 5: An example of mPnP-LLM’s modality adaptation for the multimodal QA task between two
input modalities, namely RGB camera view and LiDAR point cloud. The text tokenizer, detokenizer,
input and output embedding layers of the LLM are omitted for simplicity.
4
Figure 13:Example of modal adaptation of mPnP-LLM to
two input modes. Reproduced with permission from [139].
Copyright 2023, licensed under CC BY 4.0, ACM.
LLM
DecoderMultimodal Encoder
Text
Audio
Image
Video
Feed Forward
Video Q-former
Input text
Feed ForwardCross attention
Feed Forward
'
mh
txthSelf attention1.  Grill the tomatoes in a pan
2.  Cook bacon
3.  Place the tomato and bacon 
on the breadRobot Action Sequence
Figure 14:The schematic diagram of AVBLIP.
injection without full model fine-tuning. These systems
automatically adjust modality weights based on environ-
mental conditions. Graph Neural Networks offer structural
fusion capabilities by representing multimodal elements as
graphnodeswithspatial/semanticedges.SystemslikeGRID
[84] and VeriGraph [85] employ Graph Attention Networks
to combine visual scene graphs with verbal commands,
significantly enhancing spatial reasoning for robotic tasks.
7.2. Technological Evolution and Practical
Applications
With the continuous evolution of artificial intelligence
models,servicerobotshavebecomeakeyareafortheappli-
cationofartificialintelligencetechnology.Withitsexcellent
perception and reasoning efficiency, MLLMs, as the core
intelligent center of service robots, play an indispensable
role in action command generation and task planning. This
sectionwilldiscussindepthfrommultipledimensionssuch
as model evaluation optimization, robot system integration,
and task planning interaction, and systematically analyze
the innovative practices and technological breakthroughs of
MLLMs in service robot task planning scenarios.
In the field of model evaluation and optimization, re-
searchers are committed to improving the performance and
efficiencyofMLLMs.TheOmniBenchbenchmarkproposed
by Li et al. [139] aims to comprehensively evaluate the
parallel processing and reasoning ability of the model for
multimodal inputs such as visual, auditory, and text. By
constructing a standardized evaluation system, it provides
an important basis for quantifying the performance of the
model. Aiming at the problem of insufficient multimodal
processing efficiency in resource-constrained scenarios, the
mPnP-LLM model developed by Huang et al. [140] cre-
atively combines single-modal encoders with dynamically
configurable LLM modules and supports dynamic training
at runtime, as shown in Figure 13. This design enables the
model to independently select effective modes according to
actual needs, which greatly improves resource utilization
efficiency.Inthefaceoftheapplicationbottleneckcausedbythemodel’sdependenceonspecificinputformats,theLLM-
Bind framework proposed by Zhu et al. [141] transforms
multimodal input into task-specific token sequences with
the help of expert mixing (MoE) mechanism, and realizes
efficient processing and output conversion of multimodal
data such as image, text, video and audio. In addition, the
RespLLM framework proposed by Zhang et al. [142], an-
otherwaytounifytextandaudiorepresentations,isapplied
torespiratoryhealthpredictiontasks.Theframeworkmakes
full use of the rich prior knowledge of pre-trained LLM,
and realizes the deep fusion of audio and text information
through cross-modal attention mechanism, which provides
a new technical paradigm for multimodal model to process
heterogeneous data.
At the application level of robot systems, many studies
havefocusedonthedeepintegrationofMLLMsintoservice
robot systems. The service robot system constructed by
Ni et al. [143] integrates hardware design, scene modeling
algorithm, mobile navigation strategy based on TEB path
planning, grasping operation based on pre-training model,
and task planning module based on visual language model,
forming a complete technical chain and realizing the whole
process intelligence from perception to decision-making.
The multimodal language model agent system proposed by
Chung et al. [144] combines the task planning function
withtheinteractionabilityofthephysicalrobotbyintegrat-
ing the router, the chat robot and the task planner, which
significantly improves the decision-making efficiency and
interaction flexibility of the robot in practical applications.
The research work of Jiang et al. [145] proposes an innova-
tive multimodal prompt formula, which transforms diversi-
fied robot operation tasks into a unified sequence modeling
problem, and establishes an evaluation protocol including
multimodal tasks and system generalization ability, which
provides a new direction for the standardization research of
robot task planning.
Intermsoftaskplanningandenvironmentalinteraction,
researchers continue to explore how to make robots bet-
ter adapt to complex environments. The Inner Monologue
mechanism proposed by Huang et al. [146], by integrating
multimodal environmental feedback information, can gen-
erate a task planning scheme that is more suitable for the
actual scene, effectively coordinate the input information of
differentsensors,andenhancetheAI’sabilitytounderstand
and respond to the environment. The system developed by
Liu et al. [147] gives full play to the advantages of text
understanding and visual processing of MLLMs, and can
accuratelytransformuserdialogueandenvironmentalvisual
information into a robot executable operation plan. The
framework proposed by Wang et al. [148], with the help
of multimodal GPT-4V, deeply integrates natural language
instructionswithrobotvisualperception,whichsignificantly
improves the accuracy and flexibility of embodied mission
planning.ThePaLM-Eembodiedlanguagemodelproposed
byDriessetal.[149]realizesthedirectconnectionbetween
sensor data and language model in the real world through
staggered vision, continuous state estimation and text input
First Author et al.:Preprint submitted to ElsevierPage 15 of 24

LLM-based task planning for service robots: A review
coding, and builds a bridge between perceptual information
and semantic understanding. As shown in Figure 14, the
AVBLIPmodelproposedbyKambaraetal.[150],byintro-
ducing a multimodal encoder, supports joint input of video,
audio, voice and text, and can efficiently generate robot
action sequences. The multimodal interaction framework
proposedbyLaietal.[151],integratesvoicecommandsand
postureinformation,combinestheglobalinformationofthe
environment obtained by the visual system, uses a large-
scalelanguagemodeltoanalyzethespeechtextandbound-
ing box data, and avoids the model to generate unreason-
able output through key control syntax constraints, thereby
generating reliable robot action sequences. In addition, the
service robot task reasoning mechanism proposed by Tian
et al. [152], innovatively integrates multimodal information
and ontology knowledge, systematically manages user and
environmentinformationthroughontologyknowledgebase,
collects and integrates multi-modal data such as vision,
speechandsceneknowledgeinrealtime,andconductsdeep
reasoningbasedonfine-tuningLLM,whichprovidesricher
and more reliable knowledge support for robot decision-
making.
In terms of closed-loop integration, systems like Inner
Monologue [144] demonstrate how continuous visual feed-
back from RGB-D sensors can be integrated with visual-
language models to dynamically update task plans when
environmental changes are detected. For human-robot col-
laboration scenarios, approaches such as AudioPaLM [112]
enable natural language corrections during task execution,
allowing users to verbally adjust the robot’s actions when
deviations occur. The integration of reinforcement learning
from human feedback (RLHF [53]) provides another im-
portant mechanism for adapting LLM outputs to real-world
uncertainties, where human demonstrations and corrections
helpalignthemodel’splanningwithphysicalconstraintsand
taskrequirements.Thesedevelopmentshighlightthecritical
need to move beyond static, open-loop task specifications
andinsteaddevelopsystemsthatcanmaintainplanfeasibil-
ity through continuous perception-action cycles and human
interaction.
Furthermore, the recent breakthroughs in embodied in-
telligence, exemplified by Google’s Gemini Robotics [153]
and NVIDIA’s GR00T [154], represent a paradigm shift
from traditional LLMs by fundamentally bridging the gap
betweencognitiveAIandphysicalroboticexecution.Unlike
conventional LLMs, which are confined to textual data and
generate abstract outputs like language or code without di-
rect interaction with the physical world, these novel models
are designed as Vision-Language-Action (VLA) systems
that integrate perception, understanding, and action into
a cohesive whole. Google’s Gemini Robotics [153], built
upon the Gemini 2.0 [155] model, operates as an on-device
VLAarchitecture,processingreal-timesensorinputssuchas
RGB-D and LiDAR to directly output executable joint con-
trol commands, thereby bypassing intermediate symbolic
planning and significantly reducing latency for enhanced
real-timeperformanceandprivacy.Concurrently,NVIDIA’sGR00T [154] employs a dual-system architecture where a
vision-languagemoduleinterpretstheenvironmentbeforea
diffusion transformer generates fluid motor actions, further
distinguishedbyitsuseofsyntheticdatagenerationthrough
GR00T-Dreams, which compresses skill acquisition from
months to 36 hours, and its seamless Sim2Real transfer
via Isaac Sim 5.0. This architectural and methodological
innovation allows both systems to transcend the limita-
tions of traditional cloud-dependent LLMs by embedding
physical feasibility checks and uncertainty-aware planning
directly into their decision-making processes. The result is
an embodied intelligence capable of sensing the physical
environment,interpretinghigh-leveltaskintentfromnatural
language, and translating it into safe, executable physical
actions-acriticaladvancementforapplicationsrangingfrom
industrialautomationtoserviceroboticsinunstructureden-
vironments, thereby tightly coupling the cognitive prowess
of AI with the physical dexterity of robotic systems.
8. Current Challenges and Future Prospects
From the review and analysis, it is obvious that sig-
nificant efforts have been made to improve the planning
abilityofservicerobots,andsomenotableprogresshasbeen
achieved. However, due to the complexity and diversity of
the domestic environment, task planning for service robots
stillrequiressubstantialexploration.AsshownintheFigure
15,themainproblemsinthefieldoftaskplanningofservice
robots based on LLM can be summarized as follows.
•Insufficiencyinreal-timeperformanceandsafety:The
inference processes of current LLMs often demand
considerable computational power and time, thereby
limiting the ability to guarantee real-time decision-
making.HowtorealizeefficientcomputationofLLM
is a general step towards real-time planning. Com-
pounding this, the inherent latency, coupled with the
potential for hallucinations and uncertainties during
the planning process, poses increased safety risks for
robots during task execution.
•Challengesofmultimodalfusion:Itisstillatechnical
problem to effectively fuse text, vision, speech and
other modal information and establish a reasonable
fusion strategy. How to design an efficient fusion
architectureandbalancetheweightsbetweendifferent
modalities needs further study.
•Model evaluation and optimization: The lack of eval-
uationcriteriaandtoolsformultimodalLLMmakesit
difficulttocomprehensivelyevaluatetheperformance
androbustnessofthemodel.Atthesametime,howto
optimizethemodelstructureandimprovethecompu-
tational efficiency and generalization ability is also a
problem to be solved.
•Insufficient active task cognition: At present, LLM
still has shortcomings in active task cognition. The
service robot can only perform task planning and
First Author et al.:Preprint submitted to ElsevierPage 16 of 24

LLM-based task planning for service robots: A review
Future 
Prospects
Current 
ChallengesLLM illusion and uncertainty
Challenges of multimodal fusion
The lack of  model evaluation 
Insufficient active task cognition1More powerful LLM
2Establishment and improvement of 
the model evaluation benchmark
3Breakthrough of multimodal fusion
4Multimodal -based task cognition
5Human -robot interaction to 
enhance robot task planning
Figure 15:Some current challenges and future perspectives in task planning for service robots.
execution according to the user’s instructions, and it
isdifficulttoperformautonomousreasoningandplan-
ningaccordingtoenvironmentalinformationandtask
objectives. How to construct a reasoning model with
activetaskcognitiveabilityneedsfurtherexploration.
•Tactile language models are underdeveloped: Com-
paredtoLLMs,VLMs,andMLLMs,tactilelanguage
models remain in their early stages of development.
Challengesincludescarcetactiledatasets,underdevel-
oped multimodal fusion architectures, and the need
for breakthroughs in aligning dynamic contact state
representations with language understanding.
In view of the existing problems, the proposed research
directionsofLLM-basedtaskplanningfordomesticservice
robots include:
•More powerful LLM for robot task planning: With
the continuous development of technology, the per-
formance of LLM has significantly improved [156,
155, 157]. In order to enable LLM to respond in real-
time[158,159,160],itisnecessarytoexplorevarious
strategiessuchasmodelcompressionandacceleration
techniques [161, 162], edge computing deployment
[163], and more efficient prompt engineering [32].
Consideringthesecurityoftaskplanning,recentstud-
ies have also made substantial progress in mitigating
hallucinations in LLMs [164, 165, 166]. It is also
extremely essential to ensure that the generated plans
comply with physical and safety constraints [65, 56,
66]. To further enhance their generalization ability
and robustness in robot task planning and executing
service-oriented tasks, it remains essential to incor-
porate external knowledge bases and complementary
techniques [81, 167, 168].
•Establishment and improvement of the model eval-
uation benchmark: LLMs have been widely used in
the field of robotics. Although numerous evaluation
benchmarks exist to assess their perception and plan-
ning capabilities [169, 170, 171, 172], there is a lack
of a unified and recognized benchmark for single-
modal and multimodel, so that each model data can
be quantified. Therefore, it is necessary to establishspecific evaluation benchmarks tailored to the task
planning field of service robots, such as task success
rate, planning efficiency, robustness, etc. This would
facilitateamoreaccurateandcomparativeassessment
of model performance.
•Breakthrough of multimodal fusion: The text-based
LLM has limitations in spatial perception and dy-
namicadaptation[71],whilevisual-basedandauditory-
based models have difficulties in understanding ab-
stractconceptsandextractingeffectiveinformationin
noisyenvironments,respectively[118,88].Therefore,
the effective fusion of text, visual, auditory and other
modalinformationandtheestablishmentofareason-
able fusion strategy are the key to improving the task
planningabilityofservicerobots[125,126,142].For
instance, exploring more efficient modal alignment
methodsandcross-modalattentionmechanismscould
enableeffectiveinteractionandfusionacrossdifferent
types of modal information [173, 174].
•Multimodal-based active task cognition: Numerous
studies have demonstrated the feasibility of integrat-
ingLLMwithtaskcognition[175,126,176].Through
the combination of vision, auditory and LLM, ser-
vice robots can gain stronger active task cognitive
ability,enablingthemtomakeautonomousreasoning
andplanningaccordingtoenvironmentalinformation,
sound signals and task objectives, while adapting to
the dynamic changing environments. To this end,
further research can be conducted on how to deeply
integratemultimodalinformationwithtaskcognition,
such as by developing a unified multimodal task cog-
nition model or incorporating task cognition into the
pre-training process of multimodal LLMs.
•Human-robot interaction to enhance robot task plan-
ning: Human-robot interaction is another critical av-
enue for enhancing the task planning capabilities of
servicerobots,alongsidemultimodalinformation.Be-
yond perceiving the environment through vision and
audio,servicerobotscaninferuserintentionsthrough
direct dialogue and interaction, enabling task plan-
ning that better aligns with human preferences [177,
First Author et al.:Preprint submitted to ElsevierPage 17 of 24

LLM-based task planning for service robots: A review
178,179].Byintegratingtechnologiessuchasspeech
recognition, natural language processing, and affec-
tive computing, more natural and efficient human-
robot interactions can be achieved, thereby signifi-
cantly improving the effectiveness of task planning.
•Advancing tactile intelligence in robotic systems:
Future research should focus on three key areas:
developing comprehensive tactile-linguistic datasets
[180], creating efficient architectures for real-time
multimodal fusion [181], and advancing closed-loop
tactile control in embodied LLMs[182]. These ad-
vancementswillenablemoreadaptiveroboticinterac-
tionthroughenhancedmultimodalunderstandingand
responsivetactileintegration,particularlyinprecision
tasks requiring delicate manipulation.
9. Conclusion
This paper systematically reviews the current task plan-
ning methods for service robots based on LLMs, with a
particularfocusonkeychallengesandrecentadvancements.
Specifically, the basic knowledge of LLM is introduced.
Then, from the unique perspective of input modality dif-
ference, the research trends of LLMs based on text, vision,
audio and multimodal fusion in task planning are discussed
in depth, as shown in Figure 16. Also, a comprehensive
and detailed analysis of the existing systems, frameworks
and models is carried out, as well as the bottlenecks and
problems faced by current research. Finally, constructive
and forward-looking insights are put forward on the future
research direction of this field. It is anticipated that the
findings of this paper can support the development of more
robust, reliable, adaptable, and efficient LLM-based task
planning methods for service robots, thereby promoting the
field to a new level.
CRediT authorship contribution statement
Shaohan Bian: Writing – review & editing, Writing –
original draft, Investigation, Formal analysis.Ying Zhang:
Writing – review & editing, Writing – original draft, In-
vestigation, Supervision, .Guohui Tian: Writing – review
& editing, Investigation.Zhiqiang Miao: Writing – re-
view & editing, Writing – original draft, Investigation.Ed-
mond Q. Wu: Writing – review & editing, Investigation.
Simon X. Yang: Writing – review & editing, Investigation.
Changchun Hua:Writing–review&editing,Investigation,
Supervision.
Declaration of competing interest
Theauthorsdeclarethattheyhavenoknowncompeting
financial interests or personal relationships that could have
appeared to influence the work reported in this paper.Acknowledgments
ThisworkwassupportedinpartbytheNationalNatural
ScienceFoundationofChina(62203378,and62203377),in
partbytheHebeiNaturalScienceFoundation(F2024203036,
and F2024203115), in part by the Science Research Project
of Hebei Education Department (BJK2024195), and in
part by the S&T Program of Hebei (236Z2002G, and
236Z1603G).
References
[1] Dominik Bauer, Peter Hönig, Jean-Baptiste Weibel, José García-
Rodríguez, Markus Vincze, et al. Challenges for monocular 6d
object pose estimation in robotics.IEEE Transactions on Robotics,
2024.
[2] Ying Zhang, Guohui Tian, Cui-Hua Zhang, Changchun Hua, Weili
Ding,andChoonKiAhn. Environmentmodelingforservicerobots
fromataskexecutionperspective.IEEE/CAAJournalofAutomatica
Sinica, 2025.
[3] Stefano Dafarra, Ugo Pattacini, Giulio Romualdi, Lorenzo Rapetti,
Riccardo Grieco, Kourosh Darvish, Gianluca Milani, Enrico Valli,
Ines Sorrentino, Paolo Maria Viceconte, et al. icub3 avatar system:
Enabling remote fully immersive embodiment of humanoid robots.
Science Robotics, 9(86):eadh3834, 2024.
[4] RhysNewbury,MorrisGu,LachlanChumbley,ArsalanMousavian,
ClemensEppner,JürgenLeitner,JeannetteBohg,AntonioMorales,
Tamim Asfour, Danica Kragic, et al. Deep learning approaches
to grasp synthesis: A review.IEEE Transactions on Robotics,
39(5):3994–4015, 2023.
[5] Kechun Xu, Zhongxiang Zhou, Jun Wu, Haojian Lu, Rong Xiong,
and Yue Wang. Grasp, see and place: Efficient unknown object
rearrangement with policy structure prior.IEEE Transactions on
Robotics, 2024.
[6] Ying Zhang, Guohui Tian, Xuyang Shao, Mengyang Zhang, and
Shaopeng Liu. Semantic grounding for long-term autonomy of
mobile robots toward dynamic object search in home environ-
ments.IEEE Transactions on Industrial Electronics, 70(2):1655–
1665, 2022.
[7] Manisha Natarajan and Matthew Gombolay. Trust and dependence
on robotic decision support.IEEE Transactions on Robotics, 2024.
[8] Jian Li, Yadong Mo, Shijie Jiang, Lifang Ma, Ying Zhang, and
Shimin Wei. Bathing assistive devices and robots for the elderly.
Biomimetic Intelligence and Robotics, page 100218, 2025.
[9] Ying Zhang, Guohui Tian, and Huanzhao Chen. Exploring the
cognitive process for service task in smart home: A robot service
mechanism.Future Generation Computer Systems, 102:588–602,
2020.
[10] ChenWang,DanfeiXu,andLiFei-Fei. Generalizabletaskplanning
through representation pretraining.IEEE Robotics and Automation
Letters, 7(3):8299–8306, 2022.
[11] Deshuai Zheng, Jin Yan, Tao Xue, and Yong Liu. A knowledge-
based task planning approach for robot multi-task manipulation.
Complex & Intelligent Systems, 10(1):193–206, 2024.
[12] Simon Odense, Kamal Gupta, and William G Macready. Neural-
guided runtime prediction of planners for improved motion and
task planning with graph neural networks. In2022 IEEE/RSJ
InternationalConferenceonIntelligentRobotsandSystems(IROS),
pages 12471–12478. IEEE, 2022.
[13] Patrik Haslum, Nir Lipovetzky, Daniele Magazzeni, Christian
Muise, Ronald Brachman, Francesca Rossi, and Peter Stone.An
introductiontotheplanningdomaindefinitionlanguage,volume13.
Springer, 2019.
[14] YingZhang,GuohuiTian,XuyangShao,ShaopengLiu,Mengyang
Zhang,andPengDuan. Buildingmetric-topologicalmaptoefficient
object search for mobile robot.IEEE Transactions on Industrial
Electronics, 69(7):7076–7087, 2022.
First Author et al.:Preprint submitted to ElsevierPage 18 of 24

LLM-based task planning for service robots: A review
Input ModeMultimodal -
based LLM
Task  PlanningAudio -based 
LLM Task 
PlanningVision -based 
LLM Task 
PlanningText-based LLM 
Task Planning
VLM enhances spatial perception
Task decomposition and dynamic adjustment
Active task cognitionTask decomposition
Chain reasoning
Symbolic representation
Joint modeling of speech and language
Environmental sound perception and reasoning
Interactive voice command processing
Multimodal unified spatial representation
Environmental feedback loop
Embodied reasoning and interaction
Core Architecture Method Taxonomy
Transformer 
Visual encoder
+
LLM
+
Cross-modal attention
Audio encoder
+
LLM
+
Contrastive learning
Multimodal encoder
+
Unified fusion module
Figure 16:Task planning for different input modalities.
[15] Wenrui Zhao and Weidong Chen. Hierarchical pomdp planning for
object manipulation in clutter.Robotics and Autonomous Systems,
139:103736, 2021.
[16] Shaopeng Liu, Guohui Tian, Ying Zhang, Mengyang Zhang, and
Shuo Liu. Service planning oriented efficient object search: A
knowledge-basedframeworkforhomeservicerobot.ExpertSystems
with Applications, 187:115853, 2022.
[17] Ying Zhang, Guohui Tian, Jiaxing Lu, Mengyang Zhang, and
Senyan Zhang. Efficient dynamic object search in home environ-
ment by mobile robot: A priori knowledge-based approach.IEEE
Transactions on Vehicular Technology, 68(10):9466–9477, 2019.
[18] ShilongSun,ChiyaoLi,ZidaZhao,HaodongHuang,andWenfuXu.
Leveraging large language models for comprehensive locomotion
control in humanoid robots design.Biomimetic Intelligence and
Robotics, 4(4):100187, 2024.
[19] Yeseung Kim, Dohyun Kim, Jieun Choi, Jisang Park, Nayoung Oh,
andDaehyungPark. Asurveyonintegrationoflargelanguagemod-
elswithintelligentrobots.IntelligentServiceRobotics,17(5):1091–
1107, 2024.
[20] Fanlong Zeng, Wensheng Gan, Yongheng Wang, Ning Liu, and
Philip S Yu. Large language models for robotics: A survey.arXiv
preprint arXiv:2311.07226, 2023.
[21] Peihan Li, Zijian An, Shams Abrar, and Lifeng Zhou. Large
language models for multi-robot systems: A survey.arXiv preprint
arXiv:2502.03814, 2025.
[22] Yongcheng Cui, Ying Zhang, Cui-Hua Zhang, and Simon X Yang.
Task cognition and planning for service robots.Intelligence &
Robotics, 5(1):119–142, 2025.
[23] ChristoforosMavrogiannis,FrancescaBaldini,AllanWang,Dapeng
Zhao,PeteTrautman,AaronSteinfeld,andJeanOh.Corechallenges
ofsocialrobotnavigation:Asurvey.ACMTransactionsonHuman-
Robot Interaction, 12(3):1–39, 2023.
[24] Anbalagan Loganathan and Nur Syazreen Ahmad. A systematic
review on recent advances in autonomous mobile robot navigation.
Engineering Science and Technology, an International Journal,
40:101343, 2023.
[25] Ceng Zhang, Junxin Chen, Jiatong Li, Yanhong Peng, and Zebing
Mao.Largelanguagemodelsforhuman–robotinteraction:Areview.
Biomimetic Intelligence and Robotics, 3(4):100131, 2023.[26] SherryYang,OfirNachum,YilunDu,JasonWei,PieterAbbeel,and
Dale Schuurmans. Foundation models for decision making: Prob-
lems,methods,andopportunities.arXivpreprintarXiv:2303.04129,
2023.
[27] Jiankai Sun, Chuanyang Zheng, Enze Xie, Zhengying Liu, Ruihang
Chu, Jianing Qiu, Jiaqi Xu, Mingyu Ding, Hongyang Li, Mengzhe
Geng, et al. A survey of reasoning with foundation models.arXiv
preprint arXiv:2312.11562, 2023.
[28] Yujia Qin, Shengding Hu, Yankai Lin, Weize Chen, Ning Ding,
GanquCui,ZheniZeng,XuanheZhou,YufeiHuang,ChaojunXiao,
et al. Tool learning with foundation models.ACM Computing
Surveys, 57(4):1–40, 2024.
[29] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin.
Attentionisallyouneed.Advancesinneuralinformationprocessing
systems, 30, 2017.
[30] Alec Radford, Karthik Narasimhan, Tim Salimans, Ilya Sutskever,
etal. Improvinglanguageunderstandingbygenerativepre-training.
2018.
[31] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina
Toutanova. Bert: Pre-training of deep bidirectional transformers
for language understanding. InProceedings of the 2019 conference
of the North American chapter of the association for computational
linguistics:humanlanguagetechnologies,volume1(longandshort
papers), pages 4171–4186, 2019.
[32] TomBrown,BenjaminMann,NickRyder,MelanieSubbiah,JaredD
Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam,
Girish Sastry, Amanda Askell, et al. Language models are few-
shot learners.Advances in neural information processing systems,
33:1877–1901, 2020.
[33] LongOuyang,JeffreyWu,XuJiang,DiogoAlmeida,CarrollWain-
wright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Kata-
rina Slama, Alex Ray, et al. Training language models to follow
instructions with human feedback.Advances in neural information
processing systems, 35:27730–27744, 2022.
[34] ColinRaffel,NoamShazeer,AdamRoberts,KatherineLee,Sharan
Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu.
Exploring the limits of transfer learning with a unified text-to-text
transformer.Journal of machine learning research, 21(140):1–67,
First Author et al.:Preprint submitted to ElsevierPage 19 of 24

LLM-based task planning for service robots: A review
2020.
[35] Neil Houlsby, Andrei Giurgiu, Stanislaw Jastrzebski, Bruna Mor-
rone, Quentin De Laroussilhe, Andrea Gesmundo, Mona Attariyan,
and Sylvain Gelly. Parameter-efficient transfer learning for nlp. In
International conference on machine learning, pages 2790–2799.
PMLR, 2019.
[36] Rewon Child, Scott Gray, Alec Radford, and Ilya Sutskever. Gen-
erating long sequences with sparse transformers.arXiv preprint
arXiv:1904.10509, 2019.
[37] Sam McCandlish, Jared Kaplan, Dario Amodei, and OpenAI Dota
Team. An empirical model of large-batch training.arXiv preprint
arXiv:1812.06162, 2018.
[38] Nan Du, Yanping Huang, Andrew M Dai, Simon Tong, Dmitry
Lepikhin,YuanzhongXu,MaximKrikun,YanqiZhou,AdamsWei
Yu, Orhan Firat, et al. Glam: Efficient scaling of language models
with mixture-of-experts. InInternational conference on machine
learning, pages 5547–5569. PMLR, 2022.
[39] WilliamFedus,BarretZoph,andNoamShazeer. Switchtransform-
ers: Scaling to trillion parameter models with simple and efficient
sparsity.Journal of Machine Learning Research, 23(120):1–39,
2022.
[40] Aohan Zeng, Xiao Liu, Zhengxiao Du, Zihan Wang, Hanyu Lai,
Ming Ding, Zhuoyi Yang, Yifan Xu, Wendi Zheng, Xiao Xia, et al.
Glm-130b: An open bilingual pre-trained model.arXiv preprint
arXiv:2210.02414, 2022.
[41] Aixin Liu, Bei Feng, Bin Wang, Bingxuan Wang, Bo Liu, Cheng-
gang Zhao, Chengqi Dengr, Chong Ruan, Damai Dai, Daya Guo,
et al. Deepseek-v2: A strong, economical, and efficient mixture-of-
experts language model.arXiv preprint arXiv:2405.04434, 2024.
[42] Xiao Bi, Deli Chen, Guanting Chen, Shanhuang Chen, Damai Dai,
Chengqi Deng, Honghui Ding, Kai Dong, Qiushi Du, Zhe Fu,
et al. Deepseek llm: Scaling open-source language models with
longtermism.arXiv preprint arXiv:2401.02954, 2024.
[43] Ying Zhang, Maoliang Yin, Heyong Wang, and Changchun Hua.
Cross-levelmulti-modalfeatureslearningwithtransformerforrgb-d
object recognition.IEEE Transactions on Circuits and Systems for
Video Technology, 33(12):7121–7130, 2023.
[44] Mandar Joshi, Danqi Chen, Yinhan Liu, Daniel S Weld, Luke
Zettlemoyer, and Omer Levy. Spanbert: Improving pre-training by
representing and predicting spans.Transactions of the association
for computational linguistics, 8:64–77, 2020.
[45] Li Dong, Nan Yang, Wenhui Wang, Furu Wei, Xiaodong Liu,
YuWang,JianfengGao,MingZhou,andHsiao-WuenHon. Unified
languagemodelpre-trainingfornaturallanguageunderstandingand
generation.Advancesinneuralinformationprocessingsystems,32,
2019.
[46] PanZhou,XingyuXie,ZhouchenLin,andShuichengYan. Towards
understanding convergence and generalization of adamw.IEEE
transactions on pattern analysis and machine intelligence, 2024.
[47] Noam Shazeer and Mitchell Stern. Adafactor: Adaptive learning
rates with sublinear memory cost. InInternational Conference on
Machine Learning, pages 4596–4604. PMLR, 2018.
[48] KaiLv,YuqingYang,TengxiaoLiu,QinghuiGao,QipengGuo,and
Xipeng Qiu. Full parameter fine-tuning for large language models
with limited resources.arXiv preprint arXiv:2306.09782, 2023.
[49] HaokunLiu,DerekTam,MohammedMuqeeth,JayMohta,Tenghao
Huang, Mohit Bansal, and Colin A Raffel. Few-shot parameter-
efficient fine-tuning is better and cheaper than in-context learning.
AdvancesinNeuralInformationProcessingSystems,35:1950–1965,
2022.
[50] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu,
YuanzhiLi,SheanWang,LuWang,WeizhuChen,etal. Lora:Low-
rank adaptation of large language models.ICLR, 1(2):3, 2022.
[51] Xiang Lisa Li and Percy Liang. Prefix-tuning: Optimizing continu-
ouspromptsforgeneration.arXivpreprintarXiv:2101.00190,2021.
[52] ShengyuZhang,LinfengDong,XiaoyaLi,SenZhang,XiaofeiSun,
Shuhe Wang, Jiwei Li, Runyi Hu, Tianwei Zhang, Fei Wu, et al.
Instruction tuning for large language models: A survey.arXivpreprint arXiv:2308.10792, 2023.
[53] Robert Kirk, Ishita Mediratta, Christoforos Nalmpantis, Jelena
Luketina,EricHambro,EdwardGrefenstette,andRobertaRaileanu.
Understandingtheeffectsofrlhfonllmgeneralisationanddiversity.
arXiv preprint arXiv:2310.06452, 2023.
[54] Swaroop Mishra, Daniel Khashabi, Chitta Baral, Yejin Choi, and
Hannaneh Hajishirzi. Reframing instructional prompts to gptk’s
language.arXiv preprint arXiv:2109.07830, 2021.
[55] TakeshiKojima,ShixiangShaneGu,MachelReid,YutakaMatsuo,
and Yusuke Iwasawa. Large language models are zero-shot reason-
ers.Advancesinneuralinformationprocessingsystems,35:22199–
22213, 2022.
[56] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei
Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought
prompting elicits reasoning in large language models.Advances in
neural information processing systems, 35:24824–24837, 2022.
[57] Kevin Yang, Yuandong Tian, Nanyun Peng, and Dan Klein. Re3:
Generating longer stories with recursive reprompting and revision.
arXiv preprint arXiv:2210.06774, 2022.
[58] Maciej Besta, Ales Kubicek, Roman Niggli, Robert Gerstenberger,
Lucas Weitzendorf, Mingyuan Chi, Patrick Iff, Joanna Gajda, Piotr
Nyczyk, Jürgen Müller, et al. Multi-head rag: Solving multi-aspect
problems with llms.arXiv preprint arXiv:2406.05085, 2024.
[59] Soyeong Jeong, Jinheon Baek, Sukmin Cho, Sung Ju Hwang, and
Jong C Park. Adaptive-rag: Learning to adapt retrieval-augmented
large language models through question complexity.arXiv preprint
arXiv:2403.14403, 2024.
[60] Kunal Sawarkar, Abhilasha Mangal, and Shivam Raj Solanki.
Blended rag: Improving rag (retriever-augmented generation) accu-
racy with semantic search and hybrid query-based retrievers. In
2024IEEE7thInternationalConferenceonMultimediaInformation
Processing and Retrieval (MIPR), pages 155–161. IEEE, 2024.
[61] Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and Hannaneh
Hajishirzi. Self-rag: Learning to retrieve, generate, and critique
through self-reflection. InThe Twelfth International Conference on
Learning Representations, 2023.
[62] Diji Yang, Jinmeng Rao, Kezhen Chen, Xiaoyuan Guo, Yawen
Zhang, Jie Yang, and Yi Zhang. Im-rag: Multi-round retrieval-
augmented generation through learning inner monologues. In
Proceedings of the 47th International ACM SIGIR Conference on
Research and Development in Information Retrieval, pages 730–
740, 2024.
[63] YanDing,XiaohanZhang,ChrisPaxton,andShiqiZhang. Taskand
motion planning with large language models for object rearrange-
ment. In2023 IEEE/RSJ International Conference on Intelligent
Robots and Systems (IROS), pages 2086–2092. IEEE, 2023.
[64] IshikaSingh,ValtsBlukis,ArsalanMousavian,AnkitGoyal,Danfei
Xu, Jonathan Tremblay, Dieter Fox, Jesse Thomason, and Animesh
Garg. Progprompt: Generating situated robot task plans using
large language models. In2023 IEEE International Conference on
RoboticsandAutomation(ICRA),pages11523–11530.IEEE,2023.
[65] Tom Silver, Soham Dan, Kavitha Srinivas, Joshua B Tenenbaum,
Leslie Kaelbling, and Michael Katz. Generalized planning in pddl
domains with pretrained large language models. InProceedings
of the AAAI conference on artificial intelligence, volume 38, pages
20256–20264, 2024.
[66] Xiaodong Li, Guohui Tian, and Yongcheng Cui. Fine-grained task
planningforservicerobotsbasedonobjectontologyknowledgevia
large language models.IEEE Robotics and Automation Letters,
2024.
[67] Zhenyu Wu, Ziwei Wang, Xiuwei Xu, Jiwen Lu, and Haibin Yan.
Embodiedtaskplanningwithlargelanguagemodels.arXivpreprint
arXiv:2307.01848, 2023.
[68] Allen Z Ren, Anushri Dixit, Alexandra Bodrova, Sumeet Singh,
Stephen Tu, Noah Brown, Peng Xu, Leila Takayama, Fei Xia, Jake
Varley, et al. Robots that ask for help: Uncertainty alignment for
large language model planners.arXiv preprint arXiv:2307.01928,
2023.
First Author et al.:Preprint submitted to ElsevierPage 20 of 24

LLM-based task planning for service robots: A review
[69] Yuchen Liu, Luigi Palmieri, Sebastian Koch, Ilche Georgievski,
and Marco Aiello. Delta: Decomposed efficient long-term robot
task planning using large language models.arXiv preprint
arXiv:2404.03275, 2024.
[70] Shyam Sundar Kannan, Vishnunandan LN Venkatesh, and Byung-
CheolMin. Smart-llm:Smartmulti-agentrobottaskplanningusing
largelanguagemodels. In2024IEEE/RSJInternationalConference
on Intelligent Robots and Systems (IROS), pages 12140–12147.
IEEE, 2024.
[71] Bo Liu, Yuqian Jiang, Xiaohan Zhang, Qiang Liu, Shiqi Zhang,
Joydeep Biswas, and Peter Stone. Llm+ p: Empowering large
language models with optimal planning proficiency.arXiv preprint
arXiv:2304.11477, 2023.
[72] Tomoya Kawabe, Tatsushi Nishi, Ziang Liu, and Tomofumi Fuji-
wara. Task planning for robot manipulator using natural language
task input with large language models. In2024 IEEE 20th Interna-
tionalConferenceonAutomationScienceandEngineering(CASE),
pages 3484–3489. IEEE, 2024.
[73] Chang Chen, Hany Hamed, Doojin Baek, Taegu Kang, Yoshua
Bengio, and Sungjin Ahn. Extendable long-horizon planning via
hierarchicalmultiscalediffusion.arXivpreprintarXiv:2503.20102,
2025.
[74] Lutfi Eren Erdogan, Nicholas Lee, Sehoon Kim, Suhong Moon,
Hiroki Furuta, Gopala Anumanchipalli, Kurt Keutzer, and Amir
Gholami. Plan-and-act: Improving planning of agents for long-
horizon tasks.arXiv preprint arXiv:2503.09572, 2025.
[75] Hong Cao, Rong Ma, Yanlong Zhai, and Jun Shen. Llm-collab:
a framework for enhancing task planning via chain-of-thought and
multi-agent collaboration.Applied Computing and Intelligence,
4(2):328–348, 2024.
[76] Yue Zhen, Sheng Bi, Lu Xing-tong, Pan Wei-qin, Shi Hai-peng,
Chen Zi-rui, and Fang Yi-shu. Robot task planning based on
large language model representing knowledge with directed graph
structures.arXiv preprint arXiv:2306.05171, 2023.
[77] Jeongeun Park, Seungwon Lim, Joonhyung Lee, Sangbeom Park,
MinsukChang,YoungjaeYu,andSungjoonChoi. Clara:classifying
and disambiguating user commands for reliable interactive robotic
agents.IEEE Robotics and Automation Letters, 9(2):1059–1066,
2023.
[78] Hyobin Ong, Youngwoo Yoon, Jaewoo Choi, and Minsu Jang. A
simple baseline for uncertainty-aware language-oriented task plan-
ner for embodied agents. In2024 21st International Conference on
Ubiquitous Robots (UR), pages 677–682. IEEE, 2024.
[79] Ruoyu Wang, Zhipeng Yang, Zinan Zhao, Xinyan Tong, Zhi Hong,
and Kun Qian. Llm-based robot task planning with exceptional
handling for general purpose service robots. In2024 43rd Chinese
Control Conference (CCC), pages 4439–4444. IEEE, 2024.
[80] Zonghao Mu, Wenyu Zhao, Yue Yin, Xiangming Xi, Wei Song,
Jianjun Gu, and Shiqiang Zhu. Kggpt: empowering robots with
openai’schatgptandknowledgegraph. InInternationalConference
on Intelligent Robotics and Applications, pages 340–351. Springer,
2023.
[81] Yan Ding, Xiaohan Zhang, Saeid Amiri, Nieqing Cao, Hao Yang,
AndyKaminski,ChadEsselink,andShiqiZhang. Integratingaction
knowledgeandllmsfortaskplanningandsituationhandlinginopen
worlds.Autonomous Robots, 47(8):981–997, 2023.
[82] Ying Zhang, Maoliang Yin, Wenfu Bi, Haibao Yan, Shaohan Bian,
Cui-Hua Zhang, and Changchun Hua. Zisvfm: Zero-shot object
instance segmentation in indoor robotic environments with vision
foundationmodels.IEEETransactionsonRobotics,41:1568–1580,
2025.
[83] Guoqin Tang, Qingxuan Jia, Zeyuan Huang, Gang Chen, Ning Ji,
and Zhipeng Yao. 3d-grounded vision-language framework for
robotic task planning: Automated prompt synthesis and supervised
reasoning.arXiv preprint arXiv:2502.08903, 2025.
[84] Zhe Ni, Xiaoxin Deng, Cong Tai, Xinyue Zhu, Qinghongbing Xie,
Weihang Huang, Xiang Wu, and Long Zeng. Grid: Scene-graph-
based instruction-driven robotic task planning. In2024 IEEE/RSJInternationalConferenceonIntelligentRobotsandSystems(IROS),
pages 13765–13772. IEEE, 2024.
[85] Daniel Ekpo, Mara Levy, Saksham Suri, Chuong Huynh, and Ab-
hinavShrivastava. Verigraph:Scenegraphsforexecutionverifiable
robot planning.arXiv preprint arXiv:2411.10446, 2024.
[86] Aoran Mei, Guo-Niu Zhu, Huaxiang Zhang, and Zhongxue Gan.
Replanvlm: Replanning robotic tasks with visual language models.
IEEE Robotics and Automation Letters, 2024.
[87] Chenguang Huang, Oier Mees, Andy Zeng, and Wolfram Burgard.
Visual language maps for robot navigation. In2023 IEEE Inter-
national Conference on Robotics and Automation (ICRA), pages
10608–10615. IEEE, 2023.
[88] KeisukeShirai,CristianCBeltran-Hernandez,MasashiHamaya,At-
sushi Hashimoto, Shohei Tanaka, Kento Kawaharazuka, Kazutoshi
Tanaka, Yoshitaka Ushiku, and Shinsuke Mori. Vision-language
interpreter for robot task planning. In2024 IEEE International
ConferenceonRoboticsandAutomation(ICRA),pages2051–2058.
IEEE, 2024.
[89] Zhirong Luan, Yujun Lai, Rundong Huang, Shuanghao Bai, Yuedi
Zhang, Haoran Zhang, and Qian Wang. Enhancing robot task
planning and execution through multi-layer large language models.
Sensors, 24(5):1687, 2024.
[90] JiayangHuang,ChristianLimberg,SyedMuhammadNashitArshad,
Qifeng Zhang, and Qiang Li. Combining vlm and llm for enhanced
semanticobjectperceptioninrobotichandovertasks. In2024WRC
Symposium on Advanced Robotics and Automation (WRC SARA),
pages 135–140. IEEE, 2024.
[91] Jiaqi Zhang, Zinan Wang, Jiaxin Lai, and Hongfei Wang. Gptarm:
An autonomous task planning manipulator grasping system based
on vision–language models.Machines, 13(3):247, 2025.
[92] Naoki Wake, Atsushi Kanehira, Kazuhiro Sasabuchi, Jun Taka-
matsu,andKatsushiIkeuchi.Gpt-4v(ision)forrobotics:Multimodal
task planning from human demonstration.IEEE Robotics and
Automation Letters, 2024.
[93] Sichao Liu, Jianjing Zhang, Robert X Gao, Xi Vincent Wang, and
Lihui Wang. Vision-language model-driven scene understanding
and robotic object manipulation. In2024 IEEE 20th International
ConferenceonAutomationScienceandEngineering(CASE),pages
21–26. IEEE, 2024.
[94] ShaopengLiu,GuohuiTian,XuyangShao,andShuoLiu. Behavior
cloning-based robot active object detection with automatically gen-
erated data and revision method.IEEE Transactions on Robotics,
39(1):665–680, 2022.
[95] Shaocong Mo, Ming Cai, Lanfen Lin, Ruofeng Tong, Qingqing
Chen, Fang Wang, Hongjie Hu, Yutaro Iwamoto, Xian-Hua Han,
and Yen-Wei Chen. Mutual information-based graph co-attention
networks for multimodal prior-guided magnetic resonance imaging
segmentation.IEEETransactionsonCircuitsandSystemsforVideo
Technology, 32(5):2512–2526, 2021.
[96] An-An Liu, Hongshuo Tian, Ning Xu, Weizhi Nie, Yongdong
Zhang, and Mohan Kankanhalli. Toward region-aware attention
learning for scene graph generation.IEEE Transactions on Neural
Networks and Learning Systems, 33(12):7655–7666, 2021.
[97] Mengyang Zhang, Guohui Tian, Ying Zhang, and Hong Liu. Se-
quential learning for ingredient recognition from images.IEEE
Transactions on Circuits and Systems for Video Technology,
33(5):2162–2175, 2022.
[98] Jiayuan Xie, Wenhao Fang, Yi Cai, Qingbao Huang, and Qing Li.
Knowledge-based visual question generation.IEEE Transactions
on Circuits and Systems for Video Technology, 32(11):7547–7558,
2022.
[99] Yu Hao, Fan Yang, Nicholas Fang, and Yu-Shen Liu. Embosr: Em-
bodiedspatialreasoningforenhancedsituatedquestionansweringin
3dscenes.In2024IEEE/RSJInternationalConferenceonIntelligent
Robots and Systems (IROS), pages 9811–9816. IEEE, 2024.
[100] An-Chieh Cheng, Hongxu Yin, Yang Fu, Qiushan Guo, Ruihan
Yang, Jan Kautz, Xiaolong Wang, and Sifei Liu. Spatialrgpt:
Grounded spatial reasoning in vision language models.arXiv
First Author et al.:Preprint submitted to ElsevierPage 21 of 24

LLM-based task planning for service robots: A review
preprint arXiv:2406.01584, 2024.
[101] Jingyuan Yang, Xinbo Gao, Leida Li, Xiumei Wang, and Jinshan
Ding. Solver: Scene-object interrelated visual emotion reasoning
network.IEEE Transactions on Image Processing, 30:8686–8701,
2021.
[102] Weiwei Gu, Anant Sah, and Nakul Gopalan. Interactive visual task
learning for robots. InProceedings of the AAAI Conference on
Artificial Intelligence, volume 38, pages 10297–10305, 2024.
[103] Ruben Mascaro and Margarita Chli. Scene representations for
roboticspatialperception.AnnualReviewofControl,Robotics,and
Autonomous Systems, 8, 2024.
[104] Henry Senior, Gregory Slabaugh, Shanxin Yuan, and Luca Rossi.
Graph neural networks in vision-language image understanding: a
survey.The Visual Computer, 41(1):491–516, 2025.
[105] ZiyuanJiao,YidaNiu,ZeyuZhang,Song-ChunZhu,YixinZhu,and
Hangxin Liu. Sequential manipulation planning on scene graph. In
2022 IEEE/RSJ International Conference on Intelligent Robots and
Systems (IROS), pages 8203–8210. IEEE, 2022.
[106] FranklinKenghaghoKenfack,FerozAhmedSiddiky,FerencBalint-
Benczedi, and Michael Beetz. Robotvqa—a scene-graph-and deep-
learning-based visual question answering system for robot manipu-
lation. In2020 IEEE/RSJ International Conference on Intelligent
Robots and Systems (IROS), pages 9667–9674. IEEE, 2020.
[107] ShaoweiWang,LinglingZhang,LongjiZhu,TaoQin,Kim-HuiYap,
Xinyu Zhang, and Jun Liu. Cog-dqa: Chain-of-guiding learning
with large language models for diagram question answering. In
Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition, pages 13969–13979, 2024.
[108] Benjamin Elizalde, Soham Deshmukh, Mahmoud Al Ismail, and
HuamingWang.Claplearningaudioconceptsfromnaturallanguage
supervision. InICASSP 2023-2023 IEEE International Conference
on Acoustics, Speech and Signal Processing (ICASSP), pages 1–5.
IEEE, 2023.
[109] Andrey Guzhov, Federico Raue, Jörn Hees, and Andreas Dengel.
Audioclip: Extending clip to image, text and audio. InICASSP
2022-2022IEEEInternationalConferenceonAcoustics,Speechand
Signal Processing (ICASSP), pages 976–980. IEEE, 2022.
[110] ZhifengKong,ArushiGoel,RohanBadlani,WeiPing,RafaelValle,
and Bryan Catanzaro. Audio flamingo: A novel audio language
model with few-shot learning and dialogue abilities.arXiv preprint
arXiv:2402.01831, 2024.
[111] Zhehuai Chen, He Huang, Andrei Andrusenko, Oleksii Hrinchuk,
Krishna C Puvvada, Jason Li, Subhankar Ghosh, Jagadeesh Balam,
and Boris Ginsburg. Salm: Speech-augmented language model
with in-context learning for speech recognition and translation.
InICASSP 2024-2024 IEEE International Conference on Acous-
tics, Speech and Signal Processing (ICASSP), pages 13521–13525.
IEEE, 2024.
[112] PaulKRubenstein,ChulayuthAsawaroengchai,DucDungNguyen,
AnkurBapna,ZalánBorsos,FélixdeChaumontQuitry,PeterChen,
Dalia El Badawy, Wei Han, Eugene Kharitonov, et al. Audiopalm:
A large language model that can speak and listen.arXiv preprint
arXiv:2306.12925, 2023.
[113] SiyinWang,Chao-HanYang,JiWu,andChaoZhang. Canwhisper
perform speech-based in-context learning? InICASSP 2024-2024
IEEE International Conference on Acoustics, Speech and Signal
Processing (ICASSP), pages 13421–13425. IEEE, 2024.
[114] Soham Deshmukh, Benjamin Elizalde, Rita Singh, and Huaming
Wang. Pengi: An audio language model for audio tasks.Advances
in Neural Information Processing Systems, 36:18090–18108, 2023.
[115] Ho-HsiangWu,PremSeetharaman,KundanKumar,andJuanPablo
Bello. Wav2clip: Learning robust audio representations from clip.
InICASSP2022-2022IEEEInternationalConferenceonAcoustics,
Speech and Signal Processing (ICASSP), pages 4563–4567. IEEE,
2022.
[116] Yuan Gong, Hongyin Luo, Alexander H Liu, Leonid Karlinsky,
and James Glass. Listen, think, and understand.arXiv preprint
arXiv:2305.10790, 2023.[117] Sreyan Ghosh, Sonal Kumar, Ashish Seth, Chandra Kiran Reddy
Evuru, Utkarsh Tyagi, S Sakshi, Oriol Nieto, Ramani Duraiswami,
andDineshManocha.Gama:Alargeaudio-languagemodelwithad-
vanced audio understanding and complex reasoning abilities.arXiv
preprint arXiv:2406.11768, 2024.
[118] XingpengSun,HaomingMeng,SouradipChakraborty,AmritSingh
Bedi,andAniketBera. Beyondtext:Utilizingvocalcuestoimprove
decision making in llms for robot navigation tasks.arXiv preprint
arXiv:2402.03494, 2024.
[119] Rohan Anil, Andrew M Dai, Orhan Firat, Melvin Johnson, Dmitry
Lepikhin, Alexandre Passos, Siamak Shakeri, Emanuel Taropa,
Paige Bailey, Zhifeng Chen, et al. Palm 2 technical report.arXiv
preprint arXiv:2305.10403, 2023.
[120] Zalán Borsos, Raphaël Marinier, Damien Vincent, Eugene
Kharitonov,OlivierPietquin,MattSharifi,DominikRoblek,Olivier
Teboul, David Grangier, Marco Tagliasacchi, et al. Audiolm: a
language modeling approach to audio generation.IEEE/ACM
transactions on audio, speech, and language processing, 31:2523–
2533, 2023.
[121] SSakshi,UtkarshTyagi,SonalKumar,AshishSeth,Ramaneswaran
Selvakumar, Oriol Nieto, Ramani Duraiswami, Sreyan Ghosh, and
DineshManocha. Mmau:Amassivemulti-taskaudiounderstanding
and reasoning benchmark.arXiv preprint arXiv:2410.19168, 2024.
[122] Leonard Salewski, Stefan Fauth, A Koepke, and Zeynep Akata.
Zero-shotaudiocaptioningwithaudio-languagemodelguidanceand
audio context keywords.arXiv preprint arXiv:2311.08396, 2023.
[123] Yuchen Hu, Chen Chen, Chengwei Qin, Qiushi Zhu, Eng Siong
Chng, and Ruizhe Li. Listen again and choose the right answer: A
newparadigmforautomaticspeechrecognitionwithlargelanguage
models.arXiv preprint arXiv:2405.10025, 2024.
[124] Chao Ji, Diyuan Liu, Wei Gao, and Shiwu Zhang. Learning-based
locomotion control fusing multimodal perception for a bipedal hu-
manoid robot.Biomimetic Intelligence and Robotics, page 100213,
2025.
[125] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech,
IainBarr,YanaHasson,KarelLenc,ArthurMensch,KatherineMil-
lican, Malcolm Reynolds, et al. Flamingo: a visual language model
for few-shot learning.Advances in neural information processing
systems, 35:23716–23736, 2022.
[126] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-
2: Bootstrapping language-image pre-training with frozen image
encoders and large language models. InInternational conference
on machine learning, pages 19730–19742. PMLR, 2023.
[127] Zhengyuan Yang, Linjie Li, Jianfeng Wang, Kevin Lin, Ehsan
Azarnasab, Faisal Ahmed, Zicheng Liu, Ce Liu, Michael Zeng,
and Lijuan Wang. Mm-react: Prompting chatgpt for multimodal
reasoning and action.arXiv preprint arXiv:2303.11381, 2023.
[128] Renrui Zhang, Jiaming Han, Chris Liu, Peng Gao, Aojun Zhou,
Xiangfei Hu, Shilin Yan, Pan Lu, Hongsheng Li, and Yu Qiao.
Llama-adapter: Efficient fine-tuning of language models with zero-
init attention.arXiv preprint arXiv:2303.16199, 2023.
[129] Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed El-
hoseiny. Minigpt-4:Enhancingvision-languageunderstandingwith
advancedlargelanguagemodels.arXivpreprintarXiv:2304.10592,
2023.
[130] HaotianLiu,ChunyuanLi,QingyangWu,andYongJaeLee. Visual
instruction tuning.Advances in neural information processing
systems, 36:34892–34916, 2023.
[131] HaotianLiu,ChunyuanLi,YuhengLi,andYongJaeLee. Improved
baselines with visual instruction tuning. InProceedings of the
IEEE/CVF Conference on Computer Vision and Pattern Recogni-
tion, pages 26296–26306, 2024.
[132] Zhiliang Peng, Wenhui Wang, Li Dong, Yaru Hao, Shaohan
Huang, Shuming Ma, and Furu Wei. Kosmos-2: Grounding
multimodal large language models to the world.arXiv preprint
arXiv:2306.14824, 2023.
[133] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze
Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, et al.
First Author et al.:Preprint submitted to ElsevierPage 22 of 24

LLM-based task planning for service robots: A review
Qwen2-vl: Enhancing vision-language model’s perception of the
world at any resolution.arXiv preprint arXiv:2409.12191, 2024.
[134] Qinghao Ye, Haiyang Xu, Guohai Xu, Jiabo Ye, Ming Yan, Yiyang
Zhou, Junyang Wang, Anwen Hu, Pengcheng Shi, Yaya Shi, et al.
mplug-owl: Modularization empowers large language models with
multimodality.arXiv preprint arXiv:2304.14178, 2023.
[135] Bo Li, Yuanhan Zhang, Liangyu Chen, Jinghao Wang, Fanyi Pu,
JingkangYang,ChunyuanLi,andZiweiLiu.Mimic-it:Multi-modal
in-context instruction tuning.arXiv preprint arXiv:2306.05425,
2023.
[136] Ji Qi, Ming Ding, Weihan Wang, Yushi Bai, Qingsong Lv, Wenyi
Hong, Bin Xu, Lei Hou, Juanzi Li, Yuxiao Dong, et al. Cogcom:
Trainlargevision-languagemodelsdivingintodetailsthroughchain
of manipulations.arXiv preprint arXiv:2402.04236, 2024.
[137] Siming Yan, Min Bai, Weifeng Chen, Xiong Zhou, Qixing Huang,
and Li Erran Li. Vigor: Improving visual grounding of large vision
language models with fine-grained reward modeling. InEuropean
Conference on Computer Vision, pages 37–53. Springer, 2024.
[138] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh,
Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell,
PamelaMishkin,JackClark,etal.Learningtransferablevisualmod-
els from natural language supervision. InInternational conference
on machine learning, pages 8748–8763. PmLR, 2021.
[139] YizhiLi,GeZhang,YinghaoMa,RuibinYuan,KangZhu,Hangyu
Guo, Yiming Liang, Jiaheng Liu, Zekun Wang, Jian Yang, et al.
Omnibench:Towardsthefutureofuniversalomni-languagemodels.
arXiv preprint arXiv:2409.15272, 2024.
[140] Kai Huang, Boyuan Yang, and Wei Gao. Modality plug-and-play:
Elastic modality adaptation in multimodal llms for embodied ai.
arXiv preprint arXiv:2312.07886, 2023.
[141] Bin Zhu, Munan Ning, Peng Jin, Bin Lin, Jinfa Huang, Qi Song,
Junwu Zhang, Zhenyu Tang, Mingjun Pan, Xing Zhou, et al. Llm-
bind:Aunifiedmodality-taskintegrationframework.arXivpreprint
arXiv:2402.14891, 2024.
[142] Yuwei Zhang, Tong Xia, Aaqib Saeed, and Cecilia Mascolo. Re-
spllm:Unifyingaudioandtextwithmultimodalllmsforgeneralized
respiratory health prediction.arXiv preprint arXiv:2410.05361,
2024.
[143] MingzeNi,GangXu,HongsenLi,ZhaomingLuo,LiminPang,and
BinchaoYu.Designofaservicerobotsystembasedonamultimodal
large model. In2024 6th International Symposium on Robotics
& Intelligent Manufacturing Technology (ISRIMT), pages 81–86.
IEEE, 2024.
[144] TongLeeChung,JianxinPang,andJunCheng. Empoweringrobots
withmultimodallanguagemodelsfortaskplanningwithinteraction.
In2024 IEEE 14th International Symposium on Chinese Spoken
Language Processing (ISCSLP), pages 358–362. IEEE, 2024.
[145] Yunfan Jiang, Agrim Gupta, Zichen Zhang, Guanzhi Wang,
Yongqiang Dou, Yanjun Chen, Li Fei-Fei, Anima Anandkumar,
Yuke Zhu, and Linxi Fan. Vima: Robot manipulation with multi-
modal prompts. 2023.
[146] WenlongHuang,FeiXia,TedXiao,HarrisChan,JackyLiang,Pete
Florence, Andy Zeng, Jonathan Tompson, Igor Mordatch, Yevgen
Chebotar, et al. Inner monologue: Embodied reasoning through
planning with language models.arXiv preprint arXiv:2207.05608,
2022.
[147] Yang Liu, Yanchao Zhao, Weichao Guo, Xinjun Sheng, and Han
Ding. Enhancing household service robots with a dual-arm mobile
manipulator and multimodal large language models. In2024 IEEE
International Conference on Robotics and Biomimetics (ROBIO),
pages 1815–1820. IEEE, 2024.
[148] Jiaqi Wang, Enze Shi, Huawen Hu, Chong Ma, Yiheng Liu, Xuhui
Wang, Yincheng Yao, Xuan Liu, Bao Ge, and Shu Zhang. Large
language models for robotics: Opportunities, challenges, and per-
spectives.Journal of Automation and Intelligence, 2024.
[149] DannyDriess,FeiXia,MehdiSMSajjadi,CoreyLynch,Aakanksha
Chowdhery, Ayzaan Wahid, Jonathan Tompson, Quan Vuong,
TianheYu,WenlongHuang,etal.Palm-e:Anembodiedmultimodallanguage model. 2023.
[150] Motonari Kambara, Chiori Hori, Komei Sugiura, Kei Ota, De-
vesh K Jha, Sameer Khurana, Siddarth Jain, Radu Corcodel, Diego
Romeres, and Jonathan Le Roux. Human action understanding-
based robot planning using multimodal llm. InIEEE International
Conference on Robotics and Automation (ICRA) Workshop, 2024.
[151] Yuzhi Lai, Shenghai Yuan, Youssef Nassar, Mingyu Fan, Atmaraaj
Gopal, Arihiro Yorita, Naoyuki Kubota, and Matthias Rätsch.
Nmm-hri: Natural multi-modal human-robot interaction with voice
and deictic posture via large language model.arXiv preprint
arXiv:2501.00785, 2025.
[152] Guohui Tian1, Jian Jiang, and Shanmei Wang. Task reasoning of
service robots with fused. InProceedings of the 3rd International
ConferenceonMachineLearning,CloudComputingandIntelligent
Mining (MLCCIM2024): Volume 1, page 347. Springer Nature.
[153] GeminiRoboticsTeam,SamindaAbeyruwan,JoshuaAinslie,Jean-
Baptiste Alayrac, Montserrat Gonzalez Arenas, Travis Armstrong,
AshwinBalakrishna,RobertBaruch,MariaBauza,MichielBlokzijl,
et al. Gemini robotics: Bringing ai into the physical world.arXiv
preprint arXiv:2503.20020, 2025.
[154] Johan Bjorck, Fernando Castañeda, Nikita Cherniadev, Xingye Da,
RunyuDing,LinxiFan,YuFang,DieterFox,FengyuanHu,Spencer
Huang, et al. Gr00t n1: An open foundation model for generalist
humanoid robots.arXiv preprint arXiv:2503.14734, 2025.
[155] Gemini Team, Petko Georgiev, Ving Ian Lei, Ryan Burnell, Li-
bin Bai, Anmol Gulati, Garrett Tanzer, Damien Vincent, Zhufeng
Pan, Shibo Wang, et al. Gemini 1.5: Unlocking multimodal un-
derstanding across millions of tokens of context.arXiv preprint
arXiv:2403.05530, 2024.
[156] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad,
Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Al-
tenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical
report.arXiv preprint arXiv:2303.08774, 2023.
[157] Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav
Pandey,AbhishekKadian,AhmadAl-Dahle,AieshaLetman,Akhil
Mathur, Alan Schelten, Alex Vaughan, et al. The llama 3 herd of
models.arXiv preprint arXiv:2407.21783, 2024.
[158] Yangqing Zheng, Shunqi Mao, Dingxin Zhang, and Weidong Cai.
Llm-enhanced rapid-reflex async-reflect embodied agent for real-
timedecision-makingindynamicallychangingenvironments.arXiv
preprint arXiv:2506.07223, 2025.
[159] Chuanneng Sun, Songjun Huang, and Dario Pompili. Llm-based
multi-agent decision-making: Challenges and future directions.
IEEE Robotics and Automation Letters, 2025.
[160] Yaran Chen, Wenbo Cui, Yuanwen Chen, Mining Tan, Xinyao
Zhang, Jinrui Liu, Haoran Li, Dongbin Zhao, and He Wang.
Robogpt: an llm-based long-term decision-making embodied agent
forinstructionfollowingtasks.IEEETransactionsonCognitiveand
Developmental Systems, 2025.
[161] Shaibal Saha and Lanyu Xu. Vision transformers on the edge:
A comprehensive survey of model compression and acceleration
strategies.Neurocomputing, page 130417, 2025.
[162] Ioannis Sarridis, Christos Koutlis, Giorgos Kordopatis-Zilos, Ioan-
nisKompatsiaris,andSymeonPapadopoulos. Indistill:Information
flow-preserving knowledge distillation for model compression. In
2025 IEEE/CVF Winter Conference on Applications of Computer
Vision (WACV), pages 9033–9042. IEEE, 2025.
[163] ChangLiuandJunZhao.Enhancingstabilityandresourceefficiency
inllmtrainingforedge-assistedmobilesystems.IEEETransactions
on Mobile Computing, 2025.
[164] Robert Friel and Atindriyo Sanyal. Chainpoll: A high effi-
cacy method for llm hallucination detection.arXiv preprint
arXiv:2310.18344, 2023.
[165] Ariana Martino, Michael Iannelli, and Coleen Truong. Knowledge
injectiontocounterlargelanguagemodel(llm)hallucination. InEu-
ropean Semantic Web Conference, pages 182–185. Springer, 2023.
[166] Jiaheng Wei, Yuanshun Yao, Jean-Francois Ton, Hongyi Guo,
Andrew Estornell, and Yang Liu. Measuring and reducing
First Author et al.:Preprint submitted to ElsevierPage 23 of 24

LLM-based task planning for service robots: A review
llm hallucination without gold-standard answers.arXiv preprint
arXiv:2402.10412, 2024.
[167] Marc Hanheide, Moritz Göbelbecker, Graham S Horn, Andrzej
Pronobis, Kristoffer Sjöö, Alper Aydemir, Patric Jensfelt, Charles
Gretton, Richard Dearden, Miroslav Janicek, et al. Robot task
planning and explanation in open and uncertain worlds.Artificial
Intelligence, 247:119–150, 2017.
[168] YuqianJiang,NickWalker,JustinHart,andPeterStone.Open-world
reasoning for service robots. InProceedings of the international
conferenceonautomatedplanningandscheduling,volume29,pages
725–733, 2019.
[169] Kun Zhou, Yutao Zhu, Zhipeng Chen, Wentong Chen, Wayne Xin
Zhao, Xu Chen, Yankai Lin, Ji-Rong Wen, and Jiawei Han. Don’t
make your llm an evaluation benchmark cheater.arXiv preprint
arXiv:2311.01964, 2023.
[170] ColinWhite,SamuelDooley,ManleyRoberts,ArkaPal,BenFeuer,
Siddhartha Jain, Ravid Shwartz-Ziv, Neel Jain, Khalid Saifullah,
Siddartha Naidu, et al. Livebench: A challenging, contamination-
free llm benchmark.arXiv preprint arXiv:2406.19314, 2024.
[171] SiyuanWang,ZhuohanLong,ZhihaoFan,ZhongyuWei,andXuan-
jingHuang. Benchmarkself-evolving:Amulti-agentframeworkfor
dynamic llm evaluation.arXiv preprint arXiv:2402.11443, 2024.
[172] YidongWang,ZhuohaoYu,ZhengranZeng,LinyiYang,Cunxiang
Wang, Hao Chen, Chaoya Jiang, Rui Xie, Jindong Wang, Xing
Xie, et al. Pandalm: An automatic evaluation benchmark for llm
instruction tuning optimization.arXiv preprint arXiv:2306.05087,
2023.
[173] Yongkang Ding, Xiaoyin Wang, Hao Yuan, Meina Qu, and Xi-
angzhou Jian. Decoupling feature-driven and multimodal fusion
attention for clothing-changing person re-identification.Artificial
Intelligence Review, 58(8):241, 2025.
[174] XiaofengHan,ShunpengChen,ZenghuangFu,ZheFeng,LueFan,
Dong An, Changwei Wang, Li Guo, Weiliang Meng, Xiaopeng
Zhang, et al. Multimodal fusion and vision-language models: A
survey for robot vision.arXiv preprint arXiv:2504.02477, 2025.
[175] Zhuo Chen, Yufeng Huang, Jiaoyan Chen, Yuxia Geng, Yin Fang,
JeffZPan,NingyuZhang,andWenZhang.Lako:Knowledge-driven
visual question answering via late knowledge-to-text injection. In
Proceedings of the 11th International Joint Conference on Knowl-
edge Graphs, pages 20–29, 2022.
[176] Bin Xiao, Haiping Wu, Weijian Xu, Xiyang Dai, Houdong Hu,
Yumao Lu, Michael Zeng, Ce Liu, and Lu Yuan. Florence-2:
Advancing a unified representation for a variety of vision tasks. In
Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition, pages 4818–4829, 2024.
[177] ZhihanLv,FabioPoiesi,QiDong,JaimeLloret,andHoubingSong.
Deep learning for intelligent human–computer interaction.Applied
Sciences, 12(22):11457, 2022.
[178] Rui Zhen, Wenchao Song, Qiang He, Juan Cao, Lei Shi, and Jia
Luo. Human-computerinteractionsystem:Asurveyoftalking-head
generation.Electronics, 12(1):218, 2023.
[179] Haonan Duan, Yifan Yang, Daheng Li, and Peng Wang. Human–
robot object handover: Recent progress and future direction.
Biomimetic Intelligence and Robotics, 4(1):100145, 2024.
[180] Fengyu Yang, Chao Feng, Ziyang Chen, Hyoungseob Park, Daniel
Wang, Yiming Dou, Ziyao Zeng, Xien Chen, Rit Gangopadhyay,
AndrewOwens,etal. Bindingtouchtoeverything:Learningunified
multimodaltactilerepresentations. InProceedingsoftheIEEE/CVF
Conference on Computer Vision and Pattern Recognition, pages
26340–26353, 2024.
[181] PengHao,ChaofanZhang,DingzheLi,XiaogeCao,XiaoshuaiHao,
Shaowei Cui, and Shuo Wang. Tla: Tactile-language-action model
for contact-rich manipulation.arXiv preprint arXiv:2503.08548,
2025.
[182] WenxuanMa,XiaogeCao,YixiangZhang,ChaofanZhang,Shaobo
Yang, Peng Hao, Bin Fang, Yinghao Cai, Shaowei Cui, and Shuo
Wang. Cltp:Contrastivelanguage-tactilepre-trainingfor3dcontact
geometry understanding.arXiv preprint arXiv:2505.08194, 2025.
First Author et al.:Preprint submitted to ElsevierPage 24 of 24