# SurgRAW: Multi-Agent Workflow with Chain-of-Thought Reasoning for Surgical Intelligence

**Authors**: Chang Han Low, Ziyue Wang, Tianyi Zhang, Zhitao Zeng, Zhu Zhuo, Evangelos B. Mazomenos, Yueming Jin

**Published**: 2025-03-13 11:23:13

**PDF URL**: [http://arxiv.org/pdf/2503.10265v1](http://arxiv.org/pdf/2503.10265v1)

## Abstract
Integration of Vision-Language Models (VLMs) in surgical intelligence is
hindered by hallucinations, domain knowledge gaps, and limited understanding of
task interdependencies within surgical scenes, undermining clinical
reliability. While recent VLMs demonstrate strong general reasoning and
thinking capabilities, they still lack the domain expertise and task-awareness
required for precise surgical scene interpretation. Although Chain-of-Thought
(CoT) can structure reasoning more effectively, current approaches rely on
self-generated CoT steps, which often exacerbate inherent domain gaps and
hallucinations. To overcome this, we present SurgRAW, a CoT-driven multi-agent
framework that delivers transparent, interpretable insights for most tasks in
robotic-assisted surgery. By employing specialized CoT prompts across five
tasks: instrument recognition, action recognition, action prediction, patient
data extraction, and outcome assessment, SurgRAW mitigates hallucinations
through structured, domain-aware reasoning. Retrieval-Augmented Generation
(RAG) is also integrated to external medical knowledge to bridge domain gaps
and improve response reliability. Most importantly, a hierarchical agentic
system ensures that CoT-embedded VLM agents collaborate effectively while
understanding task interdependencies, with a panel discussion mechanism
promotes logical consistency. To evaluate our method, we introduce
SurgCoTBench, the first reasoning-based dataset with structured frame-level
annotations. With comprehensive experiments, we demonstrate the effectiveness
of proposed SurgRAW with 29.32% accuracy improvement over baseline VLMs on 12
robotic procedures, achieving the state-of-the-art performance and advancing
explainable, trustworthy, and autonomous surgical assistance.

## Full Text


<!-- PDF content starts -->

SurgRAW: Multi-Agent Workflow with Chain of
Thought Reasoning for Surgical Intelligence
Chang Han Low1, Ziyue Wang1, Tianyi Zhang1,2, Zhitao Zeng1, Zhuo Zhu1,
Evangelos B. Mazomenos3, and Yueming Jin1B
1National University of Singapore, Singapore
ymjin@nus.edu.sg
2Bioinformatics Institute (BII), Agency for Science, Technology and Research
(A*STAR), Singapore
3Wellcome/EPSRC Centre for Interventional and Surgical Sciences (WEISS) and the
Department of Medical Physics and Biomedical Engineering, University College
London, London, UK
Abstract. Integration of Vision-Language Models (VLMs) in surgical
intelligence is hindered by hallucinations, domain knowledge gaps, and
limited understanding of task interdependencies within surgical scenes,
undermining clinical reliability. While recent VLMs demonstrate strong
general reasoning and thinking capabilities, they still lack the domain
expertise and task-awareness required for precise surgical scene inter-
pretation. Although Chain-of-Thought (CoT) can structure reasoning
more effectively, current approaches rely on self-generated CoT steps,
which often exacerbate inherent domain gaps and hallucinations. To
overcome this, we present SurgRAW, a CoT-driven multi-agent frame-
work that delivers transparent, interpretable insights for most tasks in
robotic-assisted surgery. By employing specialized CoT prompts across
five tasks: instrument recognition, action recognition, action prediction,
patient data extraction, and outcome assessment, SurgRAW mitigates
hallucinations through structured, domain-aware reasoning. Retrieval-
Augmented Generation (RAG) is also integrated to external medical
knowledge to bridge domain gaps and improve response reliability. Most
importantly, a hierarchical agentic system ensures that CoT-embedded
VLM agents collaborate effectively while understanding task interdepen-
dencies, with a panel discussion mechanism promoting logical consis-
tency. To evaluate our method, we introduce SurgCoTBench, the first
reasoning-based dataset with structured frame-level annotations. With
comprehensive experiments, we demonstrate the effectiveness of pro-
posed SurgRAW with 29.32% accuracy improvement over baseline VLMs
on 12 robotic procedures, achieving SOTA performance and advancing
explainable, trustworthy, and autonomous surgical assistance. Dataset
and code is available at https://github.com/jinlab-imvr/SurgRAW.
Keywords: SurgicalSceneUnderstanding ·Chain-of-Thought ·Agentic
Reasoning.arXiv:2503.10265v1  [cs.AI]  13 Mar 2025

2 C.H. Low et al.
1 Introduction
Surgical scene understanding is fundamental to modern surgical intelligence. It
facilitates real-time monitoring of procedures, enhances intra-operative support,
and ultimately leads to improved delivery and patient outcomes. Due to the
intrinsic complexity of surgical videos, a comprehensive analysis requires ob-
servations from multiple perspectives, encompassing actions, instruments, and
patient-related factors. In the operating room, surgical teams are assigned with
specific roles, systematically reasoning through each procedural step and collab-
oratively contributing to a detailed, overall analysis [21]. This process is labor-
intensive and time-consuming, automation of intraoperative decision via com-
puter intelligence presents considerable potential for improving efficiency [28].
Multiple AI-driven approaches [8,9,29] have been developed to address these
issues. These models tackle specific tasks within the surgical workflow analy-
sis, including workflow recognition, instrument recognition, and error detection
[25,2,3,22,8,26,6,31]. However, these methods employ end-to-end networks to
generateresultswithoutprovidingfurtherexplanations,limitinginterpretability.
Additionally,modelarchitecturesfordifferenttasksvarysignificantly,necessitat-
ing careful design and task-specific data collection for training. These challenges
hinder the development of a unified and explainable system for surgical intel-
ligence. Recently, the advancement of general Large Language Models (LLMs)
has garnered attention due to their strong reasoning capabilities [1]. LLMs can
perform various tasks based on human prompts while providing reasoning pro-
cesses and can further enhance response quality when additional information
is included in the prompts. Building on LLMs, VLMs expand their capabilities
to process visual inputs and have demonstrated remarkable performance across
diverse applications, highlighting their potential in surgical scene understanding.
However, the direct application of VLMs to the surgical domain remains
challenging : i) VLMs are highly prone to hallucinations, with responses varying
significantly depending on prompt content, even for the same question, compro-
mising their reliability. ii) Primarily trained on natural scenarios, VLMs lack
surgical knowledge, often leading to inaccurate response and further exacerbat-
ing the hallucinations [31]. iii) VLMs handle different surgical tasks indepen-
dently, ignoring their interconnections within the overall workflow. As a result,
VLMs perform poorly when the surgical tasks are given as simple multiple-
choice questions (MCQs). Some approaches [15,13] attempt to fine-tune VLMs
for specific domains; however, the surgical domain lacks the massive high-quality
data required for effective fine-tuning. To mitigate these limitations, approaches
like LLaVA-COT [30] employ LLMs to generate detailed CoT as prompts to
enhance responses with zero-shot VLMs. Nonetheless, without domain-specific
knowledge, these auto-generated CoTs remain unreliable in surgical contexts,
still requiring further research to integrate VLMs into surgical intelligence.
To address these issues, we propose a Reasoning multi- AgentWorkflow for
Surgicalintelligence,termedSurgRAW.Mirroringreal-worldsurgicalteamwork,
SurgRAW assigns different tasks in surgical scene understanding to specialized
VLM agents through hierarchical orchestrators and integrates their responses

SurgRAW: Surgical Reasoning Agentic Workflow 3
Surgical Workflow Image and Question Pair Surgeon
Comprehensive Surgical Workflow Questions OrchestratorsInstrument Recognition
What is the left instrument? 
Surgical Outcome
What is the intended goal? Action Recognition
What is the ongoing action?
Action Prediction
What is the next surgical step?
Dept. Coord.
CoT-Embedded Agents Expertise Reinforcement
Semantic
Search
Retrieval & 
Augment
Vector 
Database
Document 
Chunks
Surgical 
LibraryAction  
Interpreter
Instrument 
Specialist
Action
Predictor
Patient 
Advocate
Outcome
Analyst
Panel Discussion
Action 
Evaluator
 41. Knowledge Graph Check: 
Maps instruments to their functions.
2. Inter -Chain Coherence:  
Is inter -chain logic contradiction -free?
3. Inter -Agents Collaborative Synergy:   
Do agents reinforce each other?
Rating:  1–5.VS Dept. Head
CI Dept. Head
Chain 1: Question Analysis
Chain 2: Image Feature 
Extraction
Chain 3: Cross -Referencing 
with Visual Cues
Chain 4: Option Elimination
Chain 5: Final Answer 
VerificationChain 1: Question Decomposition
Chain 2: Image Feature Extraction
Chain 3: Addressing Sub -question
Chain 4: Cross -Referencing with 
Retrieved Domain Insights
Chain 5: Option Elimination
Chain 6: Final Answer VerificationCognitive -Inference
General CoT FrameworkVisual -Semantic
General CoT FrameworkPatient Detail
What is the patient’s age?
Fig.1: The overall structure of SurgRAW processes surgical queries through hi-
erarchical orchestrators and CoT-embedded expert agents, with RAG and panel
discussions enhancing accuracy and domain reliability.
to refine the final answer. Specifically, rather than fine-tuning VLMs or re-
lying on LLM-generated CoTs, we design task-specific CoT prompts tailored
to each task. These prompts help mitigate hallucinations by ensuring robust
and interpretable reasoning. By guiding step-by-step logical deductions, they
enhance both response accuracy and explainability. Additionally, we integrate
RAG tools to bridge the domain gap, ensuring that decisions are grounded in
professional, domain-specific surgical resources, further improving the system’s
reliability. Furthermore, SurgRAW introduces a novel panel discussion mech-
anism which initiates multi-agent collaboration and debate. By verifying the
consistency between responses and intermediate steps across different agents,
SurgRAW facilitates rich interactions among agents, enabling a more compre-
hensive analysis of surgical videos. Overall, our contribution can be summarized
as the following aspects:
•We introduce SurgRAW, the first framework capable of addressing all major
questions across the entire surgical workflow while enhancing explainability.
•Mirroring the cognitive process of surgeons, we enhance VLM agents with
task-specific CoT prompts, enabling transparent and step-by-step analysis.
•A panel discussion mechanism and a RAG module are further introduced,
enabling agents to share insights and generate reliable responses.
•We introduce a new benchmark to evaluate our framework, which is the first
benchmark to encompass all tasks within the same surgical scenario.
2 Methodology
2.1 Surgical Multi-Agent Workflow
Surgical scene understanding requires both visual-semantic perception and
cognitive-inference reasoning, each with distinct processing needs. Tasks within
the surgical workflow fall into two types: visual-semantic tasks, which involve

4 C.H. Low et al.
perceptual scene analysis for fine-grained action and instrument recognition, and
cognitive-inference tasks, which rely on procedural reasoning and deeper con-
textual understanding to support decision-making. To effectively address these
tasks, we propose SurgRAW to handle each task efficiently, as illustrated in Fig.
1. SurgRAW employs a hierarchical orchestration system, where the first orches-
trator, the Department Coordinator, first classifies each query as visual-semantic
or cognitive-inference. A secondary orchestrator, either the Visual-Semantic De-
partment Head or the Cognitive-Inference Department Head, directs the task
to the granular task-level agent. This work investigates five crucial tasks in sur-
gical workflow, containing five corresponding agents: Action Interpreter, Action
Predictor, Instrument Specialist, Outcome Analyst, and Patient Advocate.
To further enhance reasoning transparency and reliability, we integrate ex-
plicit reasoning capabilities into the core agents, transforming them into CoT-
embedded VLMagents thatgenerate transparent,evidence-basedresponses.Ad-
ditionally, an evaluator agent is designed for panel discussion, which verifies sur-
gical actions and instrument usage, facilitating complex scene interpretation in
visual-semantic tasks. Meanwhile, SurgRAW enhances cognitive-inference tasks
byincorporatingRAGtoprovidemorecontext-awarerecommendationsanrefine
decision-making. Detailed designs are introduced in the sections below.
2.2 Chain-of-Thought Prompt Generation
Surgeons approach surgical scene analysis through a structured cognitive pro-
cess, beginning with an initial assessment of the clinical question to determine
the focus of analysis. Subsequently, they identify key anatomical structures, in-
terpret functional context, and refine their understanding by integrating prior
knowledge and surgical experience to resolve ambiguity [5,16,17,18]. Our CoT
prompting mirrors this thought process, structuring surgical image analysis into
sequential reasoning steps that guide VLMs like GPT-4o [1] to emulate expert-
level decision-making [20,27]. To effectively address the diverse demands of sur-
gical reasoning, we designed five specialized, task-specific CoTs, which fall into
two broad categories, visual-semantic and cognitive-inference reasoning.
The visual-semantic tasks encompass Action Recognition and Instrument
Recognition, both of which rely on visual semantics to extract and analyze scene-
leveldetails.TheseCoTsbeginwithquestionanalysis,determiningthefocusand
localizing the corresponding targets. In Action Recognition, the model extracts
contextual elements such as objects, actions, and environment to establish a
situational understanding, while Instrument Recognition requires a finer level
of visual perception, analyzing attributes like jaw configuration, surface details,
and shaft design. Both CoTs incorporate a validation step to cross-check obser-
vations against visual cues before systematically eliminating incorrect options
and selecting the best match.
Our cognitive-inference tasks, including Surgical Plan, Surgical Outcome,
and Patient Detail, extend reasoning beyond direct image interpretation by in-
corporating procedural and contextual understanding. These tasks begin with

SurgRAW: Surgical Reasoning Agentic Workflow 5
Visual -Semantic Question (Action Recognition) Cognitive -Inference Question (Action Prediction)
<Question> After the current surgical phase occurs in 
this frame, predict the most likely next surgical step? 
(a) Dissection and mobilization of the prostate. (b) … (d)
<Knowledge Graph Consistency Check> False
Graspers: ['grasping', 'retraction', 'tool manipulation’]
<Coherence Rubrics> 2 / 5                 <Collaborative Synergy Rubrics>  1 / 5
<Criteria Failed > Detected inconsistency or weak collaboration. Triggering 
action refinement and multi -agent collaboration<CoT Reasoning> Chain 1: …
<Answer> Instrument: Graspers<CoT Reasoning> Chain 1: …
<Answer> Action: Cauterization
<CoT Reasoning> Chain 1: … Chain 2: …   <Answer> Instrument: Graspers
<Knowledge Graph Consistency Check> TRUE
Graspers: [ 'grasping ', 'retraction', 'tool manipulation’]
<Coherence Rubrics> 5 / 5                 <Collaborative Synergy Rubrics>  4 / 5
<Criteria Passed  > This refinement meets our quality thresholds. Exiting 
refinement loop early. Selected candidate #2<CoT Reasoning> Chain 1: … Chain 2: …   <Answer> Instrument: Grasping<Question> What is the most likely ongoing action of the 
surgical instrument attached to the left robotic arm? 
(a) Retraction ... (d) Grasping (e) … (g)
<General Classification> Vision -Based Classification
<Specific Agent> Multi Agent Collaboration: Instrument Specialist, 
Gesture Interpreter, Action Evaluator 
<CoT Reasoning> 
Chain 1 (Question Decomposition):  Identify current phase & next step.
Chain 2 (Image Feature Extraction): Electrocautery used to divide or 
coagulate tissue … Surrounding tissue suggests it is in pelvic region
Chain 3 (Addressing Sub -question): Dissection is occurring … Next  is 
prostate mobilization before DVC suturing or urethral anastomosis.
Chain 4 (Cross -referencing with Retrieved Domain Insights): 
Retrieved knowledge indicates that after the current surgical phase, the 
most likely next surgical step would be dissection and mobilization of the 
prostate. This aligns with the standard approach in a radical 
prostatectomy where, once vascular structures like the DVC are addressed, 
prostate mobilization follows . (COT reasoning serves as a validation mechanism 
rather than blindly accepting retrieved knowledge.)
Chain 5 (Option Elimination): Suturing & closure ruled out.
Chain 6 (Final Answer Verification):  Dissection & mobilization of prostate<Specific Agent> Action Predictor
<RAG> Querying surgical library for additional domain insights …
Retrieved Content 1: … Retrieved Content 2: … 
<Executing Agent> Retrieved content is passed to Action Predictor for 
reference and analysis<General Classification> Vision -Based Classification
Fig.2: An example chat board for SurgRAW framework, illustrates the workflow
and the response.
question decomposition, breaking the inquiry into smaller sub-questions to clar-
ify intent and guide analysis. Next, the CoT extracts relevant features from the
surgical image, identifying key visual elements such as instruments, anatomi-
cal structures, and procedural context. As reasoning progresses, the approach
taken in the subsequent chain of thought diverges based on the question type.
Surgical Plan focuses on predicting the next procedural step using both image
evidence and surgical norms, while Surgical Outcome evaluates why a step is
significant and its broader impact. Meanwhile, Patient Detail extracts demo-
graphic or status-related information, aligning observations with textual cues.
The reasoning process then advances to cross-referencing the retrieved domain
insights with procedural knowledge to ensure consistency, eliminating contra-
dictions before selecting the most supported answer. While all tasks follow this
general structured framework, their domain-specific reasoning ensures tailored
yet systematic decision-making across distinct problem types.
2.3 Expertise Reinforcement through Supplementary Modules
Panel Discussion for Visual-Semantic Tasks. Despite acquiring strong rea-
soning abilities through CoT, multi-agent collaboration can further benefit spe-
cific task-level agents to achieve a more comprehensive analysis [10,12]. Visual-
semantictasks,inparticular,demandcontextualawarenessandcross-verification
to ensure procedural accuracy. For instance, the Action Interpreter may consult
the Instrument Specialist to confirm the identity or presence of a surgical in-
strument before proceeding. To tackle the challenge, SurgRAW introduces an
Action Evaluator to ensure logical coherence and consistency through a panel
discussion mechanism. This mechanism operates from three perspectives: i) To
safeguard prediction-level consistency, the Action Evaluator integrates a knowl-
edge graph derived from the information provided by Da Vinci surgical robot’s

6 C.H. Low et al.
official specifications [7]. This graph encodes permissible instrument-action re-
lationships, allowing the Instrument Specialist and Action Interpreter to cross-
reference their predictions and maintain procedural alignment. ii) To enhance
consistency and quality at the reasoning level, the Action Evaluator employs two
additional structured rubrics: the Coherence rubric evaluates the logical align-
ment of chains of thought within a task, ensuring a reliable reasoning process.
Meanwhile, the Collaborative Synergy rubric assesses how well agents reinforce
each other’s predictions while mitigating error propagation, preventing early in-
consistencies from affecting final decisions. These evaluations foster inter-agent
synergy and further reinforce evidence-based reasoning.
RAG for Cognitive-Inference Tasks. To strengthen cognitive-inference
decision-making, SurgRAW integrates a RAG module to improve the reliabil-
ity and relevance of responses [4,11]. The Cognitive-Inference Department Head
queries repository populated with medical resources from MedlinePlus [18,19].
The retrieved content is dynamically incorporated into the corresponding CoT-
embeddedVLMagents,allowingthemtorefinetheiroutputsbycross-referencing
validated medical information. This ensures alignment with established medical
standards while reducing hallucinations. By systematically validating observa-
tions and cross-referencing evidence, RAG improves SurgRAW’s dependability,
making AI-driven surgical support more transparent and clinically actionable.
3 Experiment
Dataset. WeintroduceSurgCoTBench,areasoning-basedbenchmarkforrobotic-
assisted surgery. Built from 12 patients across prostatectomy and lobectomy, it
spans five key surgical tasks, providing frame-level coverage of the entire surgical
workflow. The dataset contains 2,277 frames and 14,176 vision-query pairs. Fol-
lowing established dataset collection methodologies [32,24,14], we sourced surgi-
cal videos from YouTube and extracted transcripts using WhisperX [23]. Videos
were then downsampled to 1 fps, selecting frames that capture key procedural
moments, such as bladder neck dissection in prostatectomy and lymph node
dissection in lobectomy. Each frame is annotated with five question types cov-
ering instrument recognition, action recognition, action prediction, patient data
extraction, and outcome assessment. To refine the dataset, Instrument Recog-
nition and Action Recognition are split into left and right perspectives, gener-
ating up to seven QA pairs per frame. GPT-4o [1] generates MCQs across all
Table 1: Results of comparison experiment (%). The best results are underlined .
Methods OverallCognitive-Inference Tasks Visual-Semantic Tasks
Act. Pred. Out.Pat.Avg.Act. Rec. Inst. Rec. Avg.
MCQ 31.17 33.0319.808.4720.4336.63 45.2941.15
LLaVa-COT [30] 37.84 32.6324.4263.8140.2936.40 34.7935.56
SurgRAW 60.49 68.9544.05100.0071.0045.30 55.6750.72

SurgRAW: Surgical Reasoning Agentic Workflow 7
Chain 1 (Question Analysis): The question asks about the ongoing action …
Chain 2 (Image Feature Extraction): The instrument appears to be holding or manipulating 
tissue. There is no visible sign of cutting, suturing, or heat application like smoke
Chain 3 (Visual -to-Text Alignment): Retraction and grasping are oriented towards holding 
or maneuvering tissue… The image shows the tool interacting with tissue but without clear 
signs of cutting, suturing, or cauterization .
Chain 4 (Option Elimination): ... Cutting: No edge of a blade or cutting motion is visible…
Chain 5 (Final Answer Verification) ：Given…, grasping aligns most closely with the 
appearance
The answer is: Option (d) Grasping.MCQ without Prompt
Tool on right arm cutting 
tissue ... The answer is: Option 
(e) CuttingAnswer with SurgCOT  CoT Prompt Answer from GPT -4o
What is the most likely 
ongoing action of the 
surgical instrument attached 
to the left robotic arm? 
(a)…(d) Grasping (e) CuttingLLaVA -COT Prompt :
<SUMMARY> Analyze 
instrument tip and interaction.
<CAPTION>   Scissor -like tool
<REASONING>  suggests cutting.
<CONCLUSION>  (e) Cutting
MCQ without Prompt
Next step after dissection: 
suturing... The answer is: 
Option (d) Suturing the DVC
After the current surgical phase 
occurs in this surgical frame, 
predict the most likely next 
surgical step. (a) Dissection and 
mobilization of the prostate. 
(b)... (d) Suturing the DVC
ACTION RECOGNITION
ACTION PREDICTION
LLaVA -COT Prompt :
<SUMMARY> Identify phase, 
predict next step.
<CAPTION>   manipulating tissue.
<REASONING>  Suturing follows.
<CONCLUSION>  (d) Suturing the 
DVCChain 1 (Question Decomposition): What step is currently being performed? What tools …
Chain 2 (Image Feature Extraction): Instruments seem to be manipulating the prostate . 
Observed  vascular structures, like the dorsal venous complex (DVC), are being dissected .
Chain 3 (Addressing Sub -question): Current stage: dissecting vascular structures. Anatomy 
involved:  visible vascularized tissue. Expected outcome: hemostatic control
Chain 4 (Cross -referencing with Retrieved Domain Insights ): Robotic prostatectomy 
follows a structured sequence:  DVC control → Prostate dissection → Anastomosis → 
Closure.  DVC suturing is ongoing, the next step is prostate dissection and 
mobilization.
Chain 5 (Option Elimination): DVC suturing an ongoing phase rather than the next one .
Chain 6 (Final Answer Verification): Based on…, suturing DVC in progress, next step… 
The answer is: Option (a) Dissection and mobilization of the prostate.What is the most likely 
surgical instrument attached 
to the left robotic arm?
(a) … (b)  Monopolar Curved 
Scissors. (c)  Needle Driver.
(d)  Forceps…
INSTRUMENT RECOGNITION Chain 1 (Question Analysis): The question asks about the instrument  on the left robotic arm
Chain 2 (Image Feature Extraction): Left robotic arm appears to hold an instrument that is 
interacting with the tissue . The tip of the instrument is partially visible…
Chain 3 (Beyond Tip Features): The visible instrument appears to have serrated jaws that
are firmly grasping tissue . The visible gripping area is textured and flat.. suggests grasping .
Chain 4 (Obscured Details Inference): The absence  of sharp elements or a curved blade 
rules out  Monopolar Curved Scissors or Cautery Hook.
Chain 5 (Functional Context): The instrument is grasping tissue firmly without cutting.
Chain 6 (Ambiguity Resolution): Monopolar scissors have a curved blade and cutting 
mechanism, neither of which is present . The instrument is holding the tissue securely, 
pointing to a forceps -type instrument.
Chain 7 (Visual -to-Text Alignment): Function involves grasping and holding soft tissue 
Chain 8 (Option Elimination): Scissors? No cutting blades . Forceps? Highly probable
Chain 9 (Final Answer Verification): The broad jaw design , coupled with the gripping 
action , confirms this identification. Hence, most accurate match remains "Forceps“.
The answer is: Option (d) Forceps.MCQ without Prompt
Instrument appears to have a 
sharp tip which is characteristic 
of scissors... The answer is: 
Option (b) Monopolar Curved 
Scissors.
LLaVA -COT Prompt :
<SUMMARY> Analyze 
instrument tip, position and 
interaction.
<CAPTION>   Pointed -tipped 
tool
<REASONING>  Shape suggests 
precision cutting.
<CONCLUSION>  (b) Monopolar 
Curved ScissorsExample MCQ Questions
Fig.3: The case study for three tasks under different prompts. Red text indicates
incorrect answers, while green text highlights correct responses.
questions, followed by manual validation to ensure accuracy. To our knowledge,
SurgCoTBench is the first dataset in the surgical domain to provide structured,
frame-level annotations across the entire procedural spectrum.
Implementation and Evaluation Metrics. All experiments were conducted
using the paid API of GPT-4o in a zero-shot setting, with no task-specific fine-
tuning. Since all questions are MCQs, evaluation is based on accuracy, measured
as the percentage of correctly answered questions.
3.1 Comparisons with State-of-the-art Methods
In the surgical domain, benchmarks and CoT prompting methods are largely
absent. Therefore, we compare SurgRAW with two settings: directly querying
GPT-4o with the MCQs and using LLaVA-CoT prompt, the previous SOTA
method,togenerateCoTforpromptingGPT-4o.AsshowninTable1,MCQand
the addition of LLaVA-CoT prompt exhibit poor performance due to their lack
of structured reasoning and domain-specific knowledge. In contrast, SurgRAW
achieves a substantial improvement of 22.65% over LLaVA-CoT and 29.32%
over MCQ in overall accuracy. Notably, SurgRAW demonstrates exceptional
performance in cognitive-inference tasks, achieving 100% accuracy in Patient

8 C.H. Low et al.
Table 2: Ablation study results (%).
"Avg." means average, and "PD." for
panel discussion.
Settings Cognitive-Inference Tasks
COT RAG Act. Pred. Out. Pat. Avg.
33.03 19.80 8.47 20.43
✓ 38.74 34.52 58.45 43.90
✓ 48.53 43.56 80.76 57.62
✓ ✓ 68.95 44.05 100.00 71.00
Settings Visual-Semantic Tasks
COT PD Act. Rec. Inst. Rec. Avg.
36.63 45.29 41.15
✓ 38.26 45.29 41.93
✓ 39.51 55.67 47.95
✓ ✓ 45.30 55.67 50.72Fig.4: The comparison with tradi-
tional VQA method.
Information and a 36.32% gain in Action Prediction over LLaVA-CoT. These
results highlight the progress of our SurgRAW framework.
We also present qualitative results in Fig. 3, demonstrating that SurgRAW
not only enhances response accuracy but also provides structured explanations.
We further compare SurgRAW with Surgical-VQA [25], the SOTA VQA method
in the surgical domain, using a 3:1 split for training and test data. Surgical-VQA
is trained on the training set, and both settings are evaluated on the test set.
As shown in Fig. 4, VLMs pe rform significantly worse than Surgical-VQA when
answering MCQs, while SurgRAW significantly surpasses Surgical-VQA.
3.2 Ablation Study
Since CoT enables all task-level agents, while RAG contributes exclusively to
cognitive-inference tasks and panel discussion occurs only in visual-semantic
tasks, we conduct separate ablation studies for these two task types to eval-
uate the individual contributions of the key components in SurgRAW. As shown
in Table 2, the second raw demonstrates the significant impact of CoT prompt-
ing, improving average performance by 37.19% and 6.80% for cognitive-inference
and visual-semantic tasks, respectively. These results underscore the crucial role
of logical inference in complex tasks such as Surgical scene understanding. In
cognitive-inference tasks, RAG provides some improvement, though the best
performance is achieved when RAG is combined with CoT, suggesting that re-
trieval is further enhanced by structured problem-solving. For visual-semantic
tasks, panel discussion significantly improves action recognition performance.
Moreover, when agents are equipped with CoTs, inter-agent collaboration is fur-
ther optimized, leading to additional performance gains.

SurgRAW: Surgical Reasoning Agentic Workflow 9
4 Conclusion and Future Works
We introduced SurgRAW, the first agentic system that can solve most prob-
lems within the surgical workflow. Our results demonstrate that CoT prompting
is the key driver that enhances the accuracy of both cognitive-inference and
visual-semantic tasks. RAG and panel discussion provide additional support be-
sides CoT. SurgRAW outperforms previous methods, setting a new benchmark
for surgical understanding. Future work will focus on expanding the dataset to
improve generalization, optimizing real-time performance for surgical assistance,
and exploring CoT prompting for dynamic reasoning. These advancements will
further enhance SurgRAW’s reliability and impact in robotic-assisted surgery.
References
1. Achiam, J., Adler, S., Agarwal, S., Ahmad, L., Akkaya, I., Aleman, F.L., Almeida,
D., Altenschmidt, J., Altman, S., Anadkat, S., et al.: Gpt-4 technical report. arXiv
preprint arXiv:2303.08774 (2023)
2. Chen, K., Du, Y., You, T., Islam, M., Guo, Z., Jin, Y., Chen, G., Heng, P.A.: Llm-
assisted multi-teacher continual learning for visual question answering in robotic
surgery. In: 2024 IEEE International Conference on Robotics and Automation
(ICRA). pp. 10772–10778. IEEE (2024)
3. Gao, X., Jin, Y., Long, Y., Dou, Q., Heng, P.A.: Trans-svnet: Accurate phase
recognition from surgical videos via hybrid embedding aggregation transformer.
In: Medical Image Computing and Computer Assisted Intervention–MICCAI 2021:
24th International Conference, Strasbourg, France, September 27–October 1, 2021,
Proceedings, Part IV 24. pp. 593–603. Springer (2021)
4. Gao, Y., Xiong, Y., Gao, X., Jia, K., Pan, J., Bi, Y., Dai, Y., Sun, J., Wang, H.,
Wang, H.: Retrieval-augmented generation for large language models: A survey.
arXiv preprint arXiv:2312.10997 2(2023)
5. Hashimoto, D.A., Rosman, G., Rus, D., Meireles, O.R.: Artificial intelligence in
surgery: promises and perils. Annals of surgery 268(1), 70–76 (2018)
6. Holm, F., Ghazaei, G., Czempiel, T., Özsoy, E., Saur, S., Navab, N.: Dynamic
scene graph representation for surgical video. In: Proceedings of the IEEE/CVF
international conference on computer vision. pp. 81–87 (2023)
7. Intuitive Surgical, I.: Da vinci instruments (2024), https://www.intuitive.com/
en-us/products-and-services/da-vinci/instruments , accessed: 27-Feb-2025
8. Jin, Y., Dou, Q., Chen, H., Yu, L., Qin, J., Fu, C.W., Heng, P.A.: Sv-rcnet: work-
flow recognition from surgical videos using recurrent convolutional network. IEEE
transactions on medical imaging 37(5), 1114–1126 (2017)
9. Jin, Y., Li, H., Dou, Q., Chen, H., Qin, J., Fu, C.W., Heng, P.A.: Multi-task
recurrent convolutional network with correlation loss for surgical video analysis.
Medical image analysis 59, 101572 (2020)
10. Kim, Y., Park, C., Jeong, H., Chan, Y.S., Xu, X., McDuff, D., Lee, H., Ghassemi,
M., Breazeal, C., Park, H., et al.: Mdagents: An adaptive collaboration of llms for
medical decision-making. Advances in Neural Information Processing Systems 37,
79410–79452 (2025)
11. Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Küttler, H.,
Lewis, M., Yih, W.t., Rocktäschel, T., et al.: Retrieval-augmented generation for

10 C.H. Low et al.
knowledge-intensive nlp tasks. Advances in neural information processing systems
33, 9459–9474 (2020)
12. Li, B., Yan, T., Pan, Y., Luo, J., Ji, R., Ding, J., Xu, Z., Liu, S., Dong, H., Lin, Z.,
et al.: Mmedagent: Learning to use medical tools with multi-modal agent. arXiv
preprint arXiv:2407.02483 (2024)
13. Li, C., Wong, C., Zhang, S., Usuyama, N., Liu, H., Yang, J., Naumann, T.,
Poon, H., Gao, J.: Llava-med: Training a large language-and-vision assistant for
biomedicine in one day. Advances in Neural Information Processing Systems 36,
28541–28564 (2023)
14. Li,J.,Skinner,G.,Yang,G.,Quaranto,B.R.,Schwaitzberg,S.D.,Kim,P.C.,Xiong,
J.: Llava-surg: towards multimodal surgical assistant via structured surgical video
learning. arXiv preprint arXiv:2408.07981 (2024)
15. Lin, B., Ye, Y., Zhu, B., Cui, J., Ning, M., Jin, P., Yuan, L.: Video-llava: Learn-
ing united visual representation by alignment before projection. arXiv preprint
arXiv:2311.10122 (2023)
16. Liu, Z., Chen, K., Wang, S., Xiao, Y., Zhang, G.: Deep learning in surgical pro-
cess modeling: A systematic review of workflow recognition. Journal of Biomedical
Informatics p. 104779 (2025)
17. Manning, D.: Cognitive factors in reading medical images: thinking processes in
image interpretation. The handbook of medical image perception and techniques
pp. 107–120 (2009)
18. of Medicine (US). Friends, N.L.: MedlinePlus, vol. 5. National Institutes of Health
and the Friends of the National Library of ... (2006)
19. Miller, N., Lacroix, E.M., Backus, J.E.: Medlineplus: building and maintaining the
national library of medicine’s consumer health web service. Bulletin of the Medical
Library Association 88(1), 11 (2000)
20. Nori, H., King, N., McKinney, S.M., Carignan, D., Horvitz, E.: Capabilities of
gpt-4 on medical challenge problems. arXiv preprint arXiv:2303.13375 (2023)
21. Padoy, N.: Machine and deep learning for workflow recognition during surgery.
Minimally Invasive Therapy & Allied Technologies 28(2), 82–90 (2019)
22. Psychogyios, D., Colleoni, E., Van Amsterdam, B., Li, C.Y., Huang, S.Y., Li, Y.,
Jia, F., Zou, B., Wang, G., Liu, Y., et al.: Sar-rarp50: Segmentation of surgical
instrumentation and action recognition on robot-assisted radical prostatectomy
challenge. arXiv preprint arXiv:2401.00496 (2023)
23. Radford,A.,Kim,J.W.,Xu,T.,Brockman,G.,McLeavey,C.,Sutskever,I.:Robust
speech recognition via large-scale weak supervision. In: International conference on
machine learning. pp. 28492–28518. PMLR (2023)
24. Schmidgall, S., Kim, J.W., Jopling, J., Krieger, A.: General surgery vision trans-
former: A video pre-trained foundation model for general surgery. arXiv preprint
arXiv:2403.05949 (2024)
25. Seenivasan,L.,Islam,M.,Krishna,A.K.,Ren,H.:Surgical-vqa:Visualquestionan-
swering in surgical scenes using transformer. In: International Conference on Med-
ical Image Computing and Computer-Assisted Intervention. pp. 33–43. Springer
(2022)
26. Shao, Z., Xu, J., Stoyanov, D., Mazomenos, E.B., Jin, Y.: Think step by step:
Chain-of-gesture prompting for error detection in robotic surgical videos. IEEE
Robotics and Automation Letters (2024)
27. Sivarajkumar, S., Kelley, M., Samolyk-Mazzanti, A., Visweswaran, S., Wang, Y.:
An empirical evaluation of prompting strategies for large language models in zero-
shot clinical natural language processing: algorithm development and validation
study. JMIR Medical Informatics 12, e55318 (2024)

SurgRAW: Surgical Reasoning Agentic Workflow 11
28. Soares, A.S., Chand, M.: Why we should be sharing our operations: a game theo-
retic analysis of surgical learning. medRxiv pp. 2025–02 (2025)
29. Twinanda, A.P., Shehata, S., Mutter, D., Marescaux, J., De Mathelin, M., Padoy,
N.:Endonet:adeeparchitectureforrecognitiontasksonlaparoscopicvideos.IEEE
transactions on medical imaging 36(1), 86–97 (2016)
30. Xu, G., Jin, P., Hao, L., Song, Y., Sun, L., Yuan, L.: Llava-o1: Let vision language
models reason step-by-step. arXiv preprint arXiv:2411.10440 (2024)
31. Yuan, K., Navab, N., Padoy, N., et al.: Procedure-aware surgical video-language
pretraining with hierarchical knowledge augmentation. Advances in Neural Infor-
mation Processing Systems 37, 122952–122983 (2025)
32. Yuan, K., Srivastav, V., Yu, T., Lavanchy, J.L., Mascagni, P., Navab, N., Padoy,
N.: Learning multi-modal representations by watching hundreds of surgical video
lectures. arXiv preprint arXiv:2307.15220 (2023)