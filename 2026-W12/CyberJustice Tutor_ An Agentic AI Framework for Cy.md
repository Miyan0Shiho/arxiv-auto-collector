# CyberJustice Tutor: An Agentic AI Framework for Cybersecurity Learning via Think-Plan-Act Reasoning and Pedagogical Scaffolding

**Authors**: Baiqiang Wang, Yan Bai, Juan Li

**Published**: 2026-03-19 04:04:57

**PDF URL**: [https://arxiv.org/pdf/2603.18470v1](https://arxiv.org/pdf/2603.18470v1)

## Abstract
The integration of Large Language Models (LLMs) into cybersecurity education for criminal justice professionals is currently hindered by the "statelessness" of reactive chatbots and the risk of hallucinations in high-stakes legal contexts. To address these limitations, we propose the CyberJustice Tutor, an educational dialogue system powered by an Agentic AI framework. Unlike reactive chatbots, our system employs a "Think-Plan-Act" cognitive cycle, enabling autonomous goal decomposition, longitudinal planning, and dynamic context maintenance. We integrate a Pedagogical Scaffolding Layer grounded in Vygotsky's Zone of Proximal Development (ZPD), which dynamically adapts instructional support based on the learner's real-time progress. Furthermore, an Adaptive Retrieval Augmented Generation (RAG) core anchors the agent's reasoning in verified curriculum materials to ensure legal and technical accuracy. A comprehensive user study with 123 participants, including students, educators, and active law enforcement officers, validated the system's efficacy. Quantitative results demonstrate high user acceptance for Response Speed (4.7/5), Ease of Use (4.4/5), and Accuracy (4.3/5). Qualitative feedback indicates that the agentic architecture is perceived as highly effective in guiding learners through personalized paths, demonstrating the feasibility and usability of agentic AI for specialized professional education.

## Full Text


<!-- PDF content starts -->

CyberJustice Tutor: An Agentic AI Framework for
Cybersecurity Learning via Think–Plan–Act
Reasoning and Pedagogical Scaffolding *
Baiqiang Wang
School of Engineering and Technology
University of Washington Tacoma
Tacoma, US
wbq@uw.eduYan Bai
School of Engineering and Technology
University of Washington Tacoma
Tacoma, US
yanb@uw.eduJuan Li
Department of Computer Science
North Dakota State University
Fargo, US
j.li@ndsu.edu
Abstract—The integration of Large Language Models (LLMs)
into cybersecurity education for criminal justice professionals is
currently hindered by the ”statelessness” of reactive chatbots
and the risk of hallucinations in high-stakes legal contexts. To
address these limitations, we propose the CyberJustice Tutor,
an educational dialogue system powered by an Agentic AI
framework. Unlike reactive chatbots, our system employs a
”Think–Plan–Act” cognitive cycle, enabling autonomous goal
decomposition, longitudinal planning, and dynamic context main-
tenance. We integrate a Pedagogical Scaffolding Layer grounded
in Vygotsky’s Zone of Proximal Development (ZPD), which
dynamically adapts instructional support based on the learner’s
real-time progress. Furthermore, an Adaptive Retrieval Aug-
mented Generation (RAG) core anchors the agent’s reasoning
in verified curriculum materials to ensure legal and technical
accuracy. A comprehensive user study with 123 participants, in-
cluding students, educators, and active law enforcement officers,
validated the system’s efficacy. Quantitative results demonstrate
high user acceptance for Response Speed (4.7/5), Ease of Use
(4.4/5), and Accuracy (4.3/5). Qualitative feedback indicates that
the agentic architecture is perceived as highly effective in guiding
learners through personalized paths, demonstrating the feasibility
and usability of agentic AI for specialized professional education.
Index Terms—Agentic AI, Intelligent Tutoring Systems, Cy-
bersecurity Education
I. INTRODUCTION
The integration of Artificial Intelligence (AI) into educa-
tional curriculums has shifted from a novelty to a necessity.
In the specialized field of Cybersecurity for Criminal Jus-
tice, the demand for accessible, accurate, and context-aware
operational knowledge is critical. While Large Language
Models (LLMs) like ChatGPT have democratized access to
information, their unguided application in education presents
significant challenges. Recent longitudinal studies indicate that
uncontrolled access to generic LLMs can detrimentally affect
students’ long-term learning outcomes and exam performance
[1]. Generic models often function as passive information re-
trieval tools, suffering from statelessness—a lack of continuity
This research was supported, in part, by National Science Foundation (NSF)
Grants 2334196 and 2334197.that prevents them from guiding learners through complex,
longitudinal cognitive trajectories. Furthermore, the specific
needs of criminal justice professionals, ranging from students
focusing on conceptual legal theories to police officers requir-
ing immediate operational procedures, cannot be adequately
met by a “one-size-fits-all” reactive model.
The core challenge, therefore, is to leverage AI capabilities
to maximize pedagogical effectiveness. Achieving this goal
requires a system that moves beyond simple reactivity. It
demands a robust architecture that ensures domain accuracy
to prevent hallucinations, while simultaneously implementing
sophisticated strategies like scaffolding and reflective learn-
ing—essential components for fostering deep comprehension
[2], [3].
To address these gaps, we propose the CyberJustice Tutor,
a novel educational dialogue system designed specifically for
the intersection of cybersecurity and criminal justice. Unlike
standard chatbots, our system is built upon an Agentic AI
framework. Central to this architecture is an autonomous
pedagogical agent that employs a “Think–Plan–Act” cogni-
tive cycle. This allows the system to analyze user intent,
autonomously decompose high-level learning goals into struc-
tured plans, and execute dynamic pedagogical strategies. We
introduce a Pedagogical Scaffolding Layer grounded in Vy-
gotsky’s Zone of Proximal Development (ZPD), which adapts
the instructional support level based on the user’s real-time
progress. Furthermore, to mitigate the risk of misinformation
in high-stakes legal contexts, we implement an Adaptive
Retrieval Augmented Generation (RAG) core, grounding the
agent’s reasoning in verified curriculum materials.
This paper presents the design and evaluation of the Cyber-
Justice Tutor. Our contributions are as follows:
1)Agentic Cognitive Architecture:We introduce a novel
framework that utilizes a cognitive cycle (Think-Plan-
Act) to enable autonomous goal decomposition and
longitudinal planning, transcending the limitations of
reactive LLMs.arXiv:2603.18470v1  [cs.HC]  19 Mar 2026

2)Pedagogical Alignment with ZPD:We demonstrate the
implementation of dynamic scaffolding strategies within
an AI agent, ensuring interactions are cognitively aligned
with the learner’s developmental zone.
3)Empirical Evaluation:We report on a comprehensive
user study with novice learners and experts, providing
empirical evidence of the system’s usability, acceptance,
and perceived pedagogical utility, thereby validating the
potential of agentic architectures in specialized educa-
tional contexts.
II. RELATEDWORK
The architecture of the CyberJustice Tutor is situated at the
convergence of generative artificial intelligence in education,
the paradigm of agentic systems, and established cognitive
learning theories.
A. Generative AI and Flipped Learning in Education
The integration of Large Language Models (LLMs) into ed-
ucational curricula represents a paradigm shift from static con-
tent delivery to dynamic interaction. Recent empirical research
by Brito et al. [4] demonstrated that integrating AI-focused
modules into cybersecurity curricula could increase student
knowledge retention by 30%. This supports the ”Flipped
Classroom” model, where direct instruction shifts to pre-class
engagement. While Akc ¸ayır and Akc ¸ayır [5] noted that flipped
learning enhances engagement, it historically suffered from
a lack of immediate, personalized feedback. To address this,
the field has moved towards AI-driven tutoring. However,
unguided reliance on generic models presents significant risks.
Ohm et al. [1] found that uncontrolled ChatGPT usage corre-
lated with lower exam scores due to surface-level processing.
Furthermore, as highlighted by Bengio et al. [6] and Kasneci
et al. [7], generic models often hallucinate or lack domain-
specific grounding. To bridge this gap, effective systems must
emulate the ”Socratic” method utilized by expert human tutors
[8], prompting deep reasoning rather than merely providing
answers.
B. The Shift from AI Agents to Agentic AI
Our system architecture is informed by the rapidly evolving
taxonomy of autonomous systems. Recent surveys by Wang
et al. [9] and Xi et al. [10] distinguish between traditional
AI Agents and Agentic AI, which exhibits system-level in-
telligence through autonomous goal decomposition and per-
sistent memory. Early iterations, such as AutoGPT [11] and
BabyAGI [12], demonstrated autonomous task execution but
struggled with stability. Current frameworks emphasize multi-
agent orchestration. Wu et al. [13] introduced AutoGen for
conversational programming, while Li et al. [14] proposed
the CAMEL framework to reduce hallucinations through role-
playing. Similarly, Qian et al. [15] developed ChatDev to
simulate standardized operating procedures (SOPs) through
decentralized collaboration. Wang et al. [16] further pushed
the boundary with V oyager, demonstrating lifelong learning in
embodied agents. Our CyberJustice Tutor adopts thisAgenticAIparadigm [17], employing a unified cognitive workflow
to handle the complex reasoning required in criminal justice
scenarios.
C. Cognitive Architectures: Reasoning and RAG
To ensure reliability in high-stakes domains, robust cogni-
tive architectures are essential. The ReAct framework by Yao
et al. [18] synergizes reasoning and acting, a principle central
to our ”Think–Plan–Act” cycle. This is complemented by
Chain-of-Thought (CoT) prompting [19], which significantly
improves reasoning performance. Kojima et al. [20] further
demonstrated that LLMs are ”zero-shot reasoners” when prop-
erly prompted. However, to mitigate the critical bottleneck
of hallucination [21], RAG conceptualized by Lewis et al.
[22], has become standard. Simoni et al. [23] utilized RAG
in the MoRSE chatbot to anchor cybersecurity answers in
verified databases. Additionally, Schick et al. [24] showed that
LLMs can autonomous invoke external tools to verify facts.
By combining CoT reasoning with RAG, our system ensures
pedagogical guidance is legally accurate.
D. Pedagogical Scaffolding and Persona Design
Finally, the efficacy of an educational agent depends on its
pedagogical grounding. Our approach is rooted in scaffolding,
originally defined by Wood, Bruner, and Ross [25]. Zhang et
al. [2] validated this in the AI context with ”PromptTutor,”
proving that scaffolding strategies improve learning outcomes
compared to direct answers. This aligns with Wiggins and
McTighe’s [26] backward design framework, mirrored by our
agent’s planning module. This support is enhanced by effective
persona design. Park et al. [27] demonstrated that agents with
specific personas can simulate believable social behaviors.
In education, studies by Lieb and Goel [28] and Lo et al.
[29] indicate that consistent personas increase engagement.
Furthermore, motivation is critical; as outlined in the MSLQ
by Pintrich et al. [30], intrinsic value is a key driver of
learning. Our system integrates these findings by adopting a
Senior Cybercrime Analyst persona that socially mediates the
learning process.
III. METHODOLOGY
The design of the CyberJustice Tutor represents a con-
vergence of advanced computational intelligence and socio-
cultural educational theory. Our methodology moves beyond
the paradigm of reactive chatbots to establish an Agentic
AI framework capable of autonomous planning and dynamic
pedagogical intervention. This section delineates the theoret-
ical foundations of Agentic AI and Vygotskian Scaffolding,
followed by their technical realization in our unified system
architecture.
A. Agentic AI Framework and Cognitive Planning
The architectural foundation of the CyberJustice Tutor is
predicated on the principles of Agentic AI, a paradigm defined
by Sapkota et al. [17] as systems capable of autonomous goal
decomposition, dynamic orchestration, and persistent memory

across complex workflows. Unlike traditional AI agents that
operate as modular, task-specific entities executing isolated
operations via tool interfaces, our system leverages Agentic AI
to exhibit system-level intelligence. This distinction is critical
for educational contexts, where effective tutoring requires
not merely retrieving static information but constructing a
coherent, longitudinal learning trajectory that adapts to the
learner’s evolving needs.
To operationalize this capability, the system employs a
“Think–Plan–Act” cognitive cycle that shifts the locus of con-
trol from a linear prompt-response mechanism to an emergent,
self-correcting loop. When a novice learner poses a high-
level, unstructured query—such as asking how to investigate
a complex cybercrime scenario—the system does not simply
retrieve a generic list of steps. Instead, it utilizes its reason-
ing depth to decompose this abstract goal into a structured
pedagogical sequence, defining necessary prerequisites before
introducing advanced forensic concepts. This capacity for
multi-step reasoning allows the system to handle the ambiguity
inherent in student inquiries, bridging the gap between novice
intent and expert knowledge structure.
Furthermore, the Agentic AI framework ensures temporal
continuity and proactivity throughout the educational dialogue.
By maintaining a persistent state of the learner’s progress, the
system moves beyond passive reactivity to actively manage the
conversation flow. The architecture proactively recalls prior
misconceptions and adjusts the current pedagogical strategy
accordingly, creating a cohesive educational narrative rather
than a disjointed series of independent interactions. This
sustained context awareness is essential for mimicking the
guidance of a human mentor, ensuring that the AI’s inter-
ventions are consistently aligned with the learner’s long-term
educational objectives.
We distinguish this framework from general-purpose au-
tomation by specializing the agent’s capabilities. Rather than
prioritizing open-ended external actions (e.g., browsing the
live web or executing software commands), instead, we posi-
tion our system as exhibiting pedagogical agency. This refers
to the autonomy to alter instructional strategies without explicit
user prompting. The key novelty lies in longitudinal peda-
gogical planning—the system’s capacity for persistent goal
tracking, autonomous plan revision, and adaptive scaffolding
across multi-turn learning trajectories. In this context, the
environment the agent manipulates is not a physical interface,
but the learner’s cognitive state.
B. Pedagogical Alignment: ZPD and Instructional Scaffolding
To ensure that the autonomous capabilities of the Agentic
AI translate into effective learning outcomes, the system’s
interaction model is grounded in Vygotsky’s ZPD. As dis-
cussed by Shabani et al. [31], the ZPD represents the critical
distance between a learner’s actual developmental level, de-
termined by independent problem solving, and their potential
development under expert guidance. Within the context of
the CyberJustice Tutor, the Agentic AI assumes the role of
the “More Knowledgeable Other,” mediating this cognitivetransition. Rather than providing static information dumps
which often lead to cognitive overload, or withholding too
much information which results in frustration, the system aims
to operate precisely within this zone, offering assistance that
is contingent on the learner’s real-time performance.
This theoretical framework is operationalized through a dy-
namic strategy of instructional scaffolding, where the system
controls elements of the task initially beyond the learner’s
capacity while allowing them to concentrate on manageable
components. The interaction follows a graduated progression
of support, effectively fading assistance as the learner gains
competence.(see Table I) For novice users encountering new
concepts, the system initiates a high-support model character-
ized by direct instruction and explicit modeling of reasoning
processes. As the learner demonstrates comprehension, the
agent transitions to a collaborative mode, offering strategic
hints and partial templates that encourage the user to bridge
logical gaps independently. Finally, to verify the internalization
of concepts, the system withdraws support to pose open-ended
challenges, requiring the learner to synthesize information and
apply it autonomously to complex scenarios.
The integration of this pedagogical scaffolding with the
underlying Agentic AI architecture enables a highly responsive
educational experience. The system’s planning module con-
tinuously monitors the user’s inputs to diagnose their current
standing within the ZPD. If a learner struggles with a specific
concept, the agent detects the breakdown and dynamically
reinstates scaffolding—breaking the problem down further or
providing an analogy—before attempting to move forward
again. This adaptive calibration ensures that the instructional
pace is neither too trivial nor too overwhelming, fostering a
state of flow that is essential for mastering the intricate legal
and technical nuances of cybersecurity in criminal justice.
TABLE I
ADAPTIVESCAFFOLDINGSTAGES WITHIN THEZPD
Support Level Phase Agentic Pedagogical Strategy
High Support
(Modeling)“I Do”Direct Instruction:The agent ex-
plicitly defines concepts, models the
reasoning process, and establishes
the foundational knowledge base to
reduce cognitive load.
Guided Support
(Co-Construction)“We Do”Collaborative Inquiry:The
agent offers strategic hints, partial
templates, and guiding questions,
prompting the learner to bridge
logical gaps with assistance.
Low Support
(Autonomy)“You Do”Independent Application:The
agent withdraws scaffolding to pose
open-ended scenarios and complex
challenges, verifying internalization
and skill transfer.
C. System Architecture and Implementation
The CyberJustice Tutor is engineered as a decoupled web
application orchestrated via the LangChain framework, uti-

lizing OpenAI’s GPT-4o as the core reasoning engine for
its superior logical deduction capabilities.As illustrated in
Figure 1, upon receiving input, the agent initiates a latent
reasoning phase to classify the learner’s intent—distinguishing
between a new inquiry and a response to a prior scaffold. If
a significant knowledge gap is detected, the planning module
autonomously generates a multi-step learning path, ensuring
that the subsequent execution phase delivers content that is
structurally aligned with the user’s current Zone of Proximal
Development.
To mitigate the hallucinations common in generative models
and ensure domain accuracy, the agent’s reasoning is grounded
in a RAG pipeline. The system maintains a specialized knowl-
edge base comprising comprehensive curriculum materials,
ranging from legal statutes to digital forensics procedures.
By retrieving the top semantically relevant chunks prior to
generation, the system effectively constrains the LLM’s output
to the verified knowledge base, minimizing hallucination risks
in high-stakes contexts.
Finally, the interaction is mediated through a “Senior Cy-
bercrime Analyst” persona, designed to foster professional
immersion. To ensure the system evolves alongside its users,
a Human-in-the-Loop feedback mechanism is embedded di-
rectly into the interface. Explicit user ratings and implicit
interaction data are logged to a structured dataset, creating
a feedback loop that allows for the iterative refinement of
both the RAG retrieval relevance and the agent’s scaffolding
strategies, thereby maintaining the system’s efficacy as an
adaptive educational tool.
To comprehensively illustrate how the Agentic AI, Peda-
gogical Scaffolding, and System Architecture converge in a
real-world interaction, Figure 2 presents a detailed operational
walkthrough. This transcript highlights the agent’s internal
cognitive cycle, from decomposing the abstract topic of ‘Mal-
ware Defense’ into a structured learning plan, to applying
specific scaffolding strategies based on the user’s responses.
IV. EXPERIMENTDESIGN ANDEVALUATION
To validate the architectural integrity and pedagogical ef-
ficacy of the CyberJustice Tutor, we conducted a compre-
hensive user study focused on three critical dimensions: the
usability of the agentic interface, the factual accuracy of the
RAG-generated content, and the effectiveness of the dynamic
scaffolding strategies.
A. Participants and Study Protocol
The study cohort comprised 123 participants recruited from
Criminal Justice and Technology programs, representing a
diverse cross-section of the target user base. The demo-
graphic distribution included undergraduate CJ students (ap-
prox. 75%), active law enforcement officers, legal profession-
als, and educators. This diversity is crucial; while students
represent the ”Novice” ZPD, professionals and educators serve
as domain experts to validate the accuracy of the content.User Input & Profile
(Student / Officer)
Think Phase
Latent Reasoning
Intent Type?
Plan Generation
Create Learning PathPlan Adaptation
Diagnose &
Update Status
RAG Pipeline
Retrieve Verified ContextVector
Database
Act Phase (Execution)
Deliver Scaffolded Content
Pedagogical
ResponseNew
InquiryResponse
to Prior
Fig. 1. The architectural workflow of the CyberJustice Tutor. It classifies user
intent (Latent Reasoning), generates learning paths or analyzes gaps (Plan-
ning), and delivers verified content aligned with the user’s ZPD (Execution)
Notably, 65% of participants reported only occasional famil-
iarity with AI tools, ensuring the study measured the system’s
intuitive design rather than the users’ prior AI proficiency.
To evaluate the system in a realistic educational context,
we adopted an open-ended study protocol. Rather than en-
forcing a rigid experimental script or fixed time constraints,
participants were granted access to the CyberJustice Tutor
and encouraged to use it as a supplementary learning tool for
their coursework or professional development. Consequently,
engagement sessions were self-directed, with usage durations
varying naturally based on individual user needs. Participants
were invited to freely explore the system’s core capabilities,
such as querying foundational concepts or engaging in inves-
tigative scenarios, allowing us to assess how the agent adapts
to diverse, spontaneous user intents.
V. RESULTS
Data was collected via a post-interaction survey utiliz-
ing both Likert-scale ratings (1–5) and open-ended feedback
mechanisms to capture both the performance metrics and the
user experience of the agentic interaction.
A. Quantitative Performance
The quantitative results, summarized in Table II, indicate a
strong and consistent user acceptance of the backend agentic
intelligence across a large sample size (N=123). The inclusion
of educators and practitioners in the expanded cohort provides
significant validation of the system’s professional utility.
The system achieved exceptionally high marks forRe-
sponse Speed (4.7)andEase of Use (4.4). This performance

Case Study: Agentic Planning for Malware Defense
Context:User is identified as aCJ Student(Novice). The goal is to learn defensive strategies.
User:“What should I do to defend against malware?”
[Agent Internal State]
•Intent:New Topic (Malware Defense)→Plan Generation.
•Plan Created (5 Steps):1. Definitions & Vectors; 2. Baseline Hygiene; 3. Anti-malware Tools; 4. Layered Defenses; 5.
Incident Response.
•Current Step:Step 1 (Definitions).
•Scaffolding Strategy: High Support (“I Do”). Establish foundational concepts before tactics.
Agent:“Great question! To defend effectively, we first need to understand the threat.
Step 1: Fundamentals.Malware includes various categories likeViruses(attach to files),Worms(self-replicate), andRansomware
(encrypt data).
Check for Understanding:Can you name at least three types you’ve heard of and describe how they differ? Feel free to guess—we
will clarify misconceptions together.”
User:
“1.Virus:Attaches to a file and needs me to open it. 2.Ransomware:Locks my files until I pay money. 3.Trojan:Disguises
itself as fake software so I install it.”
[Agent Internal State]
•Intent:Response to Scaffold→Plan Adaptation.
•Evaluation:Definitions are accurate. Mastery of Step 1 core concepts confirmed.
•Next Action:Complete Step 1 (Infection Vectors) before moving to Step 2 (Hygiene).
•Scaffolding Strategy: Guided Support (“We Do”). Validate success and collaboratively fill the remaining knowledge gaps
(Worms/Rootkits).
Agent:“Spot-on! You have correctly identified the execution mechanism of a Virus versus the deception of a Trojan.
Before we move toStep 2 (Hygiene Practices), we need to cover the missing pieces:Worms(network spread) andRootkits(hidden
access). Let’s look at how these typically infect a system...”
Fig. 2. An operational instantiation of the Agentic Tutor Framework. This example illustrates how the system’sPlanning Moduletranslates a high-level user
intent (“Defend against malware”) into a structured 5-step curriculum, while theExecution Phaseapplies high-support scaffolding logic as defined in Table I.
TABLE II
USEREVALUATIONRATINGS(SCALE1–5, N=123)
Metric Mean Std. Dev Interpretation
Response Speed 4.7 0.5 Low latency routing
Ease of Use 4.4 0.7 High intuitive adoption
Accuracy 4.3 0.6 RAG effectively verifies facts
Relevance 4.2 0.7 Context-aware outputs
Practicality 4.1 0.8 Actionable guidance
Visual Appeal 3.5 1.1 Functional but needs polish
data validates the efficiency of the Agent architecture, confirm-
ing that the “Think–Plan–Act” cognitive cycle operates with-
out introducing perceptible latency. Notably, theAccuracy
(4.3)rating reinforces the effectiveness of the RAG pipeline
in mitigating hallucinations.
Regarding Visual Appeal, the moderate score (3.5±1.1)
reflects the system’s status as a functional research prototype
rather than a commercial product. The research priority was
explicitly placed on engineering the backend agentic logic
and scaffolding mechanisms. The higher standard deviation
indicates a divergence in user expectations; however, thehigh Ease of Use score suggests that while the interface
lacked aesthetic polish, it successfully facilitated the intended
pedagogical interactions without friction.
B. Qualitative Validation of Methodology
The open-ended feedback provided empirical support for the
specific pedagogical design choices embedded in the system’s
cognitive architecture.
Primary validation for theScaffolding Strategyemerged
from user comments regarding the instructional pacing. One
educator explicitly noted:“The tutor provided clear and orga-
nized explanations, and more importantly, asked me questions
in ‘next steps for you’ that inspired me to think further and
guided me to a learning flow. ”This directly confirms that
the agent successfully operated within the users’ Zone of
Proximal Development, moving beyond passive answering to
active cognitive guidance. Another participant highlighted the
system’s ability to“translate technical cybersecurity concepts
into clear, criminal justice-focused language, ”validating the
effectiveness of the Persona design in bridging domain gaps.
Furthermore, theAgentic Planning Capabilitieswere high-
lighted as a key differentiator. Users described the generated
Learning Plans as providing good structure and helping them

learn systematically. A novice user reported that the system
“helped me understand network intrusion basics by breaking
down a complex log-analysis scenario into simple, step-by-
step explanations. ”This feedback confirms that the Planning
Module successfully decomposed high-level abstract goals into
actionable educational sequences, a core capability of Agentic
AI that static chatbots often lack.
VI. CONCLUSION
This study investigated the feasibility of integrating an
Agentic AI framework into cybersecurity education for crim-
inal justice. By combining a “Think–Plan–Act” cognitive
cycle with Vygotskian scaffolding and RAG, we addressed
the limitations of generic LLMs regarding continuity and
hallucinations. Our findings from a 123-person user study
demonstrate that the CyberJustice Tutor achieves high marks
for usability and perceived accuracy. While we cannot yet
claim quantified learning gains without longitudinal testing,
the strong user acceptance and qualitative feedback suggest
that the system effectively engages learners and adapts to
their cognitive needs. This work establishes a foundational
architecture for agentic pedagogical systems, proving that
autonomous planning and dynamic scaffolding are viable and
valuable additions to AI-driven education. Future work will
focus on controlled trials to measure long-term knowledge
retention.
ACKNOWLEDGMENT
We sincerely thank criminal justice students and profession-
als for their participation in this anonymous survey on the use
of chatbots.
REFERENCES
[1] M. Ohm, C. Bungartz, F. Boes, and M. Meier, “Assessing the impact of
large language models on cybersecurity education: A study of chatgpt’s
influence on student performance,” inThe 19th International Conference
on Availability, Reliability and Security, 2024.
[2] Y . Zhang, E. L. Ouh, A. Ho, S. L. Lo, K. W. Tan, and F. Lin,
“Prompttutor: Effects of an llm-based chatbot on learning outcomes
and motivation in flipped classrooms,” inProceedings of the 30th
ACM Conference on Innovation and Technology in Computer Science
Education V . 1, 2025.
[3] T. W. Li, Y . Song, H. Sundaram, and K. Karahalios, “Can learners nav-
igate imperfect generative pedagogical chatbots? an analysis of chatbot
errors on learning,” inProceedings of the Twelfth ACM Conference on
Learning @ Scale, 2025.
[4] F. Brito, Y . Mekdad, M. Ross, M. A. Finlayson, and S. Uluagac,
“Enhancing cybersecurity education with artificial intelligence content,”
inProceedings of the 56th ACM Technical Symposium on Computer
Science Education V . 1, 2025.
[5] G. Akc ¸ayır and M. Akc ¸ayır, “The flipped classroom: A review of its
advantages and challenges,”Computers & Education, vol. 126, pp. 334–
345, 2018.
[6] Y . Bengio, G. Hinton, A. Yao, D. Song, P. Abbeel, Y . N. Harari, Y .-Q.
Zhanget al., “Managing extreme ai risks amid rapid progress,”Science,
2024.
[7] E. Kasneci, K. Seßler, S. K ¨uchemann, M. Bannert, D. Dementieva,
F. Fischer, U. Gasser, G. Groh, S. G ¨unnemann, E. H ¨ullermeieret al.,
“Chatgpt for good? on opportunities and challenges of large language
models for education,”Learning and Individual Differences, vol. 103,
p. 102274, 2023.
[8] M. T. Chi, S. A. Siler, H. Jeong, T. Yamauchi, and R. G. Hausmann,
“Learning from human tutoring,”Cognitive science, vol. 25, no. 4, pp.
471–533, 2001.[9] L. Wang, C. Ma, X. Feng, Z. Zhang, H. Yang, J. Zhang, Z. Chen,
J. Tang, X. Chen, Y . Linet al., “A survey on large language model
based autonomous agents,” inFrontiers of Computer Science, 2024.
[10] Z. Xi, W. Chen, X. Guo, W. He, Y . Ding, B. Honget al., “The rise
and potential of large language model based agents: A survey,”Science
China Information Sciences, vol. 68, no. 2, p. 121101, 2025.
[11] H. Yang, S. Yue, and Y . He, “Auto-gpt for online decision making:
Benchmarks and additional opinions,”arXiv preprint arXiv:2306.02224,
2023.
[12] Y . Nakajima, “Babyagi,”GitHub repository, 2023,
https://github.com/yoheinakajima/babyagi.
[13] Q. Wu, G. Bansal, J. Zhang, Y . Wu, B. Li, E. Zhu, L. Jiang, X. Zhang
et al., “Autogen: Enabling next-gen llm applications via multi-agent
conversation,”arXiv preprint arXiv:2308.08155, 2023.
[14] G. Li, H. Hammoud, H. Itani, D. Khizbullin, and B. Ghanem, “Camel:
Communicative agents for ”mind” exploration of large language model
society,” inAdvances in Neural Information Processing Systems, vol. 36,
2023.
[15] C. Qian, W. Liu, H. Liu, N. Chen, Y . Dang, J. Li, and C. Yang,
“Chatdev: Communicative agents for software development,”arXiv
preprint arXiv:2307.07924, 2023.
[16] G. Wang, Y . Xie, Y . Jiang, A. Mandlekar, C. Xiao, Y . Zhu, L. Fan, and
A. Anandkumar, “V oyager: An open-ended embodied agent with large
language models,”arXiv preprint arXiv:2305.16291, 2023.
[17] R. Sapkota, K. I. Roumeliotis, and M. Karkee, “Ai agents vs. agentic
ai: A conceptual taxonomy, applications and challenges,”Information
Fusion, vol. 126, p. 103599, 2025.
[18] S. Yao, J. Zhao, D. Yu, N. Du, I. Shafran, K. Narasimhan, and Y . Cao,
“React: Synergizing reasoning and acting in language models,” in
International Conference on Learning Representations (ICLR), 2023.
[19] J. Wei, X. Wang, D. Schuurmans, M. Bosma, B. Ichter, F. Xia, E. Chi,
Q. Le, and D. Zhou, “Chain-of-thought prompting elicits reasoning in
large language models,”Advances in Neural Information Processing
Systems, vol. 35, pp. 24 824–24 837, 2022.
[20] T. Kojima, S. S. Gu, M. Reid, Y . Matsuo, and Y . Iwasawa, “Large lan-
guage models are zero-shot reasoners,”Advances in neural information
processing systems, vol. 35, pp. 22 199–22 213, 2022.
[21] Z. Ji, N. Lee, R. Frieske, T. Yu, D. Su, Y . Xu, E. Ishii, Y . Bang
et al., “Survey of hallucination in natural language generation,”ACM
Computing Surveys, vol. 55, no. 12, pp. 1–38, 2023.
[22] P. Lewis, E. Perez, A. Piktus, F. Petroni, V . Karpukhin, N. Goyal,
H. K ¨uttler, M. Lewis, W.-t. Yih, T. Rockt ¨aschelet al., “Retrieval-
augmented generation for knowledge-intensive nlp tasks,”Advances in
Neural Information Processing Systems, vol. 33, pp. 9459–9474, 2020.
[23] M. Simoni, A. Saracino, V . P, and M. Conti, “Morse: Bridging the gap
in cybersecurity expertise with retrieval augmented generation,” inThe
40th ACM/SIGAPP Symposium on Applied Computing, 2025.
[24] T. Schick, J. Dwivedi-Yu, R. Dess `ı, R. Raileanu, M. Lomeli, L. Zettle-
moyer, N. Cancedda, and T. Scialom, “Toolformer: Language models
can teach themselves to use tools,”arXiv preprint arXiv:2302.04761,
2023.
[25] D. Wood, J. S. Bruner, and G. Ross, “The role of tutoring in problem
solving,”Journal of child psychology and psychiatry, vol. 17, no. 2, pp.
89–100, 1976.
[26] G. Wiggins and J. McTighe,Understanding by design. Association for
Supervision and Curriculum Development, 2005.
[27] J. S. Park, J. C. O’Brien, C. J. Cai, M. R. Morris, P. Liang, and M. S.
Bernstein, “Generative agents: Interactive simulacra of human behavior,”
inProceedings of the 36th Annual ACM Symposium on User Interface
Software and Technology, 2023, pp. 1–22.
[28] A. Lieb and T. Goel, “Student interaction with newtbot: An llm-as-tutor
chatbot for secondary physics education,” inExtended Abstracts of the
CHI Conference on Human Factors in Computing Systems, 2024.
[29] P. Y . Lo, A. Veldhuis, A. N. Antle, and S. DiPaola, “Noel: A chatbot
persona to support children designing for others,” inCHI Conference
on Human Factors in Computing Systems, 2025.
[30] P. R. Pintrich, D. A. Smith, T. Garcia, and W. J. McKeachie, “A manual
for the use of the motivated strategies for learning questionnaire (mslq),”
1991.
[31] K. Shabani, M. Khatib, and S. Ebadi, “Vygotsky’s zone of proximal
development: Instructional implications and teachers’ professional de-
velopment,”English language teaching, vol. 3, no. 4, pp. 237–248, 2010.